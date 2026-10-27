
import logging
import secrets
import string
import time as timer
from datetime import datetime, timedelta, time as datetime_time
from decimal import Decimal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database import get_connection


router = APIRouter(
    prefix="/appointments",
    tags=["Salon Appointments"],
)

logger = logging.getLogger("salon.appointments")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_mysql_time(value) -> datetime_time:
    """
    MySQL TIME values may arrive as datetime.time or timedelta.
    Normalize both formats before comparing times.
    """
    if isinstance(value, datetime_time):
        return value

    if isinstance(value, timedelta):
        total_seconds = int(value.total_seconds()) % 86400
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return datetime_time(hours, minutes, seconds)

    if isinstance(value, str):
        return datetime_time.fromisoformat(value)

    raise TypeError(
        f"Unsupported MySQL TIME value type: {type(value).__name__}"
    )


def make_reference() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "SAL-" + "".join(
        secrets.choice(alphabet) for _ in range(10)
    )


def write_log(
    cursor,
    action,
    result,
    source="api",
    appointment_id=None,
    customer_id=None,
    details=None,
    duration_ms=None,
    error_message=None,
):
    """Write an audit record using the current DB transaction."""
    import json

    cursor.execute(
        """
        INSERT INTO appointment_logs (
            appointment_id,
            customer_id,
            action,
            result,
            source,
            details,
            duration_ms,
            error_message
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            appointment_id,
            customer_id,
            action,
            result,
            source,
            json.dumps(details or {}, ensure_ascii=False),
            duration_ms,
            error_message,
        ),
    )


# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class BookingRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    phone: str = Field(min_length=7, max_length=20)
    service_id: int = Field(gt=0)
    appointment_date: str
    appointment_time: str
    branch_id: int = Field(default=1, gt=0)
    customer_address: str | None = Field(default=None, max_length=255)
    booking_source: str = "text"


# ------------------------------------------------------------
# GET: Salon services and rates
# ------------------------------------------------------------

@router.get("/services")
def list_services():
    started = timer.perf_counter()
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                service_id,
                service_name,
                description,
                price,
                duration_minutes,
                discount_percent,
                tax_percent
            FROM services
            WHERE is_active = TRUE
            ORDER BY service_name
            """
        )

        services = cursor.fetchall()

        for service in services:
            service["price"] = float(service["price"])
            service["discount_percent"] = float(
                service["discount_percent"]
            )
            service["tax_percent"] = float(service["tax_percent"])

        return {"services": services}

    except Exception:
        logger.exception("Unable to load salon services")
        raise HTTPException(
            status_code=500,
            detail="Could not load salon services",
        )

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

        logger.info(
            "SQL list-services duration_ms=%.3f",
            (timer.perf_counter() - started) * 1000,
        )


# ------------------------------------------------------------
# POST: Create a pending appointment
# ------------------------------------------------------------

@router.post("/book")
def book_appointment(payload: BookingRequest):
    started = timer.perf_counter()
    connection = None
    cursor = None

    try:
        # Validate date and time formats.
        try:
            appointment_date = datetime.strptime(
                payload.appointment_date, "%Y-%m-%d"
            ).date()

            appointment_time = datetime.strptime(
                payload.appointment_time, "%H:%M"
            ).time()

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Date must be YYYY-MM-DD and time must be HH:MM"
                ),
            ) from exc

        if appointment_date < datetime.now().date():
            raise HTTPException(
                status_code=400,
                detail="Appointment date is in the past",
            )

        full_name = payload.full_name.strip()
        phone = payload.phone.strip()

        if not full_name or not phone:
            raise HTTPException(
                status_code=400,
                detail="Customer name and phone are required",
            )

        source = (
            payload.booking_source
            if payload.booking_source in {
                "voice", "text", "website", "admin"
            }
            else "text"
        )

        connection = get_connection()
        connection.start_transaction()
        cursor = connection.cursor(dictionary=True)

        # Lock branch row during slot validation.
        cursor.execute(
            """
            SELECT branch_id, opening_time, closing_time, is_active
            FROM salon_branches
            WHERE branch_id = %s
            FOR UPDATE
            """,
            (payload.branch_id,),
        )

        branch = cursor.fetchone()

        if not branch or not branch["is_active"]:
            raise HTTPException(
                status_code=404,
                detail="Salon branch not found or inactive",
            )

        # FIX: Normalize MySQL TIME values before comparing.
        opening_time = normalize_mysql_time(
            branch["opening_time"]
        )
        closing_time = normalize_mysql_time(
            branch["closing_time"]
        )

        cursor.execute(
            """
            SELECT
                service_id,
                service_name,
                price,
                duration_minutes,
                discount_percent,
                tax_percent
            FROM services
            WHERE service_id = %s
              AND is_active = TRUE
            """,
            (payload.service_id,),
        )

        service = cursor.fetchone()

        if not service:
            raise HTTPException(
                status_code=404,
                detail="Salon service not found or inactive",
            )

        start_dt = datetime.combine(
            appointment_date, appointment_time
        )
        end_dt = start_dt + timedelta(
            minutes=int(service["duration_minutes"])
        )

        # This simple version does not support appointments crossing midnight.
        if end_dt.date() != appointment_date:
            raise HTTPException(
                status_code=400,
                detail="Appointment cannot extend past midnight",
            )

        start_time = start_dt.time()
        end_time = end_dt.time()

        if (
            start_time < opening_time
            or end_time > closing_time
        ):
            raise HTTPException(
                status_code=400,
                detail="Requested appointment is outside salon working hours",
            )

        # Prevent overlapping active reservations at this branch.
        # All bookings use the same branch lock to serialize this check.
        cursor.execute(
            """
            SELECT appointment_id
            FROM appointments
            WHERE branch_id = %s
              AND appointment_date = %s
              AND status IN ('pending_confirmation', 'confirmed')
              AND appointment_time < %s
              AND end_time > %s
            LIMIT 1
            """,
            (
                payload.branch_id,
                appointment_date,
                end_time,
                start_time,
            ),
        )

        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail=(
                    "This time slot is already reserved. "
                    "Please choose another time."
                ),
            )

        # Insert or update customer.
        cursor.execute(
            """
            INSERT INTO customers (
                full_name, phone, address_line
            )
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                full_name = VALUES(full_name),
                address_line = COALESCE(
                    VALUES(address_line), address_line
                )
            """,
            (
                full_name,
                phone,
                payload.customer_address,
            ),
        )

        cursor.execute(
            """
            SELECT customer_id
            FROM customers
            WHERE phone = %s
            """,
            (phone,),
        )

        customer = cursor.fetchone()

        if not customer:
            raise RuntimeError("Customer record could not be retrieved")

        # Calculate price using database service rates.
        price = Decimal(str(service["price"]))
        discount = (
            price
            * Decimal(str(service["discount_percent"]))
            / Decimal("100")
        )
        net_price = price - discount
        tax = (
            net_price
            * Decimal(str(service["tax_percent"]))
            / Decimal("100")
        )
        total = net_price + tax

        reference = make_reference()

        cursor.execute(
            """
            INSERT INTO appointments (
                booking_reference,
                customer_id,
                branch_id,
                appointment_date,
                appointment_time,
                end_time,
                booking_source,
                status,
                subtotal,
                discount_amount,
                tax_amount,
                total_amount
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                'pending_confirmation', %s, %s, %s, %s
            )
            """,
            (
                reference,
                customer["customer_id"],
                payload.branch_id,
                appointment_date,
                start_time,
                end_time,
                source,
                price,
                discount,
                tax,
                total,
            ),
        )

        appointment_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO appointment_services (
                appointment_id,
                service_id,
                service_name_snapshot,
                unit_price,
                quantity,
                duration_minutes,
                discount_amount,
                tax_amount,
                line_total
            )
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s)
            """,
            (
                appointment_id,
                service["service_id"],
                service["service_name"],
                price,
                int(service["duration_minutes"]),
                discount,
                tax,
                total,
            ),
        )

        cursor.execute(
            """
            INSERT INTO appointment_status_history (
                appointment_id,
                old_status,
                new_status,
                changed_by,
                reason
            )
            VALUES (
                %s, NULL, 'pending_confirmation', %s,
                'Booking created; awaiting customer confirmation'
            )
            """,
            (appointment_id, source),
        )

        elapsed_ms = (timer.perf_counter() - started) * 1000

        write_log(
            cursor,
            action="create_booking",
            result="success",
            source=source,
            appointment_id=appointment_id,
            customer_id=customer["customer_id"],
            details={
                "booking_reference": reference,
                "service": service["service_name"],
                "status": "pending_confirmation",
            },
            duration_ms=elapsed_ms,
        )

        connection.commit()

        logger.info(
            "Booking created reference=%s status=pending_confirmation "
            "sql_total_ms=%.3f",
            reference,
            elapsed_ms,
        )

        return {
            "message": (
                "Booking details saved. Confirm the details before "
                "the appointment is finalized."
            ),
            "appointment_id": appointment_id,
            "booking_reference": reference,
            "status": "pending_confirmation",
            "customer_name": full_name,
            "phone": phone,
            "service": service["service_name"],
            "date": str(appointment_date),
            "time": start_time.strftime("%H:%M"),
            "end_time": end_time.strftime("%H:%M"),
            "total_amount": round(float(total), 2),
            "sql_duration_ms": round(elapsed_ms, 3),
        }

    except HTTPException:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    except Exception as exc:
        if connection and connection.is_connected():
            connection.rollback()

        logger.exception("Appointment booking failed")

        # Log failure separately so rollback does not erase it.
        if connection and connection.is_connected():
            try:
                connection.rollback()
                cursor.close()
                cursor = connection.cursor(dictionary=True)

                write_log(
                    cursor,
                    action="create_booking",
                    result="failure",
                    source=payload.booking_source
                    if payload.booking_source in {
                        "voice", "text", "website", "admin"
                    }
                    else "text",
                    details={"branch_id": payload.branch_id},
                    duration_ms=(
                        timer.perf_counter() - started
                    ) * 1000,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                connection.commit()
            except Exception:
                logger.exception("Could not persist booking failure log")

        raise HTTPException(
            status_code=500,
            detail=(
                f"Booking failed: {type(exc).__name__}: {exc}"
            ),
        ) from exc

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

        logger.info(
            "Booking endpoint duration_ms=%.3f",
            (timer.perf_counter() - started) * 1000,
        )


# ------------------------------------------------------------
# POST: Confirm pending appointment
# ------------------------------------------------------------

@router.post("/{booking_reference}/confirm")
def confirm_appointment(booking_reference: str):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        connection.start_transaction()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT appointment_id, status, customer_id
            FROM appointments
            WHERE booking_reference = %s
            FOR UPDATE
            """,
            (booking_reference,),
        )
        appointment = cursor.fetchone()

        if not appointment:
            raise HTTPException(
                status_code=404,
                detail="Booking reference not found",
            )

        if appointment["status"] != "pending_confirmation":
            raise HTTPException(
                status_code=409,
                detail=f"Cannot confirm booking in status {appointment['status']}",
            )

        cursor.execute(
            """
            UPDATE appointments
            SET status = 'confirmed', confirmed_at = NOW()
            WHERE appointment_id = %s
            """,
            (appointment["appointment_id"],),
        )

        cursor.execute(
            """
            INSERT INTO appointment_status_history (
                appointment_id, old_status, new_status,
                changed_by, reason
            )
            VALUES (%s, 'pending_confirmation', 'confirmed',
                    'api', 'Customer confirmation received')
            """,
            (appointment["appointment_id"],),
        )

        write_log(
            cursor,
            action="confirm_booking",
            result="success",
            appointment_id=appointment["appointment_id"],
            customer_id=appointment["customer_id"],
            details={"booking_reference": booking_reference},
        )

        connection.commit()

        return {
            "message": "Appointment confirmed successfully",
            "booking_reference": booking_reference,
            "status": "confirmed",
        }

    except HTTPException:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    except Exception as exc:
        if connection and connection.is_connected():
            connection.rollback()
        logger.exception("Appointment confirmation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Confirmation failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ------------------------------------------------------------
# POST: Cancel appointment
# ------------------------------------------------------------

class CancellationRequest(BaseModel):
    reason: str = Field(default="Cancelled by customer", max_length=500)


@router.post("/{booking_reference}/cancel")
def cancel_appointment(
    booking_reference: str,
    payload: CancellationRequest,
):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        connection.start_transaction()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT appointment_id, status, customer_id
            FROM appointments
            WHERE booking_reference = %s
            FOR UPDATE
            """,
            (booking_reference,),
        )
        appointment = cursor.fetchone()

        if not appointment:
            raise HTTPException(
                status_code=404,
                detail="Booking reference not found",
            )

        if appointment["status"] not in (
            "pending_confirmation", "confirmed"
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot cancel booking in status {appointment['status']}",
            )

        old_status = appointment["status"]

        cursor.execute(
            """
            UPDATE appointments
            SET status = 'cancelled',
                cancellation_reason = %s,
                cancelled_at = NOW()
            WHERE appointment_id = %s
            """,
            (payload.reason, appointment["appointment_id"]),
        )

        cursor.execute(
            """
            INSERT INTO appointment_status_history (
                appointment_id, old_status, new_status,
                changed_by, reason
            )
            VALUES (%s, %s, 'cancelled', 'api', %s)
            """,
            (
                appointment["appointment_id"],
                old_status,
                payload.reason,
            ),
        )

        write_log(
            cursor,
            action="cancel_booking",
            result="success",
            appointment_id=appointment["appointment_id"],
            customer_id=appointment["customer_id"],
            details={
                "booking_reference": booking_reference,
                "reason": payload.reason,
            },
        )

        connection.commit()

        return {
            "message": "Appointment cancelled successfully",
            "booking_reference": booking_reference,
            "status": "cancelled",
            "reason": payload.reason,
        }

    except HTTPException:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    except Exception as exc:
        if connection and connection.is_connected():
            connection.rollback()
        logger.exception("Appointment cancellation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Cancellation failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
