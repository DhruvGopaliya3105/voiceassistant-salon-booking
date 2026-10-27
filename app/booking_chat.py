
import re
import threading
from datetime import date, datetime, timedelta

from fastapi import HTTPException

from app.routers.appointments import (
    BookingRequest,
    book_appointment,
    confirm_appointment,
    list_services,
)

_sessions = {}
_lock = threading.Lock()

YES_WORDS = {"yes", "yes please", "confirm", "confirm it", "haan", "han", "ha", "kar do", "book it"}
NO_WORDS = {"no", "no thanks", "cancel", "nahi", "nahin", "mat karo", "stop"}

BOOKING_WORDS = (
    "book appointment", "book a appointment", "book an appointment",
    "appointment book", "book salon", "salon appointment",
    "schedule appointment", "appointment chahiye", "booking karni",
    "appointment karni", "book a haircut", "book haircut",
)


def _is_yes(message):
    return message.strip().lower().strip(" .!?") in YES_WORDS


def _is_no(message):
    return message.strip().lower().strip(" .!?") in NO_WORDS


def _new_session():
    return {
        "stage": "service",
        "service_id": None,
        "service_name": None,
        "price": None,
        "full_name": None,
        "phone": None,
        "appointment_date": None,
        "appointment_time": None,
        "booking_reference": None,
        "mode": "text",
    }


def _get_services():
    result = list_services()
    return result.get("services", [])


def _find_service(message, services):
    text = message.strip().lower()

    # Accept a service ID if the user enters a number.
    if text.isdigit():
        service_id = int(text)
        for service in services:
            if service["service_id"] == service_id:
                return service

    # Match the longest service name first.
    for service in sorted(
        services,
        key=lambda item: len(item["service_name"]),
        reverse=True,
    ):
        if service["service_name"].lower() in text:
            return service

    return None


def _parse_date(message):
    text = message.strip().lower()
    today = date.today()

    if text in {"today", "aaj"}:
        return today.isoformat()
    if text in {"tomorrow", "kal"}:
        return (today + timedelta(days=1)).isoformat()

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed = datetime.strptime(message.strip(), fmt).date()
            return parsed.isoformat()
        except ValueError:
            pass

    # Support "29 October 2026" and "October 29 2026".
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            return parsed.isoformat()
        except ValueError:
            pass

    return None


def _parse_time(message):
    text = message.strip().lower().replace(".", "")
    text = re.sub(r"\s+", " ", text)

    # 2 PM, 2:30 pm, 14:30, 14:30:00
    for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text.upper(), fmt).strftime("%H:%M")
        except ValueError:
            pass

    # Accept "at 2 pm" or "time is 14:30".
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3).upper()
        try:
            return datetime.strptime(
                f"{hour}:{minute:02d} {suffix}", "%I:%M %p"
            ).strftime("%H:%M")
        except ValueError:
            return None

    return None


def _looks_like_booking(message):
    text = message.lower()
    return any(word in text for word in BOOKING_WORDS)


def handle_booking(message: str, session_id: str, mode: str = "text"):
    """
    Return a booking response when a booking conversation is active.
    Return None when the message should be handled by the general assistant.
    """
    message = message.strip()
    if not message or not session_id:
        return None

    key = session_id

    with _lock:
        state = _sessions.get(key)

        # Start a booking conversation only when the user asks for one.
        if state is None:
            if not _looks_like_booking(message):
                return None

            state = _new_session()
            state["mode"] = mode
            _sessions[key] = state
            return (
                "Of course! I can help you book a salon appointment. "
                "Which service would you like? For example, Basic Haircut, "
                "Hair Spa, Facial, or Beard Trim."
            )

        state["mode"] = mode
        lower = message.lower().strip(" .!?")

        # Let the user abandon the current draft at any point.
        if lower in {"cancel booking", "cancel this booking", "start over", "restart booking"}:
            _sessions.pop(key, None)
            return "No problem. I've cleared this booking draft. Tell me whenever you'd like to book an appointment."

        stage = state["stage"]

        if stage == "service":
            try:
                services = _get_services()
            except Exception:
                _sessions.pop(key, None)
                return "Sorry, I couldn't load the salon services from the database. Please try again."

            if not services:
                _sessions.pop(key, None)
                return "Sorry, there are currently no active salon services available."

            service = _find_service(message, services)
            if not service:
                names = ", ".join(item["service_name"] for item in services)
                return f"Please choose one of these available services: {names}."

            state["service_id"] = int(service["service_id"])
            state["service_name"] = service["service_name"]
            state["price"] = service["price"]
            state["stage"] = "name"
            return (
                f"Great choice: {service['service_name']} "
                f"(listed price ₹{service['price']}). "
                "What is your full name?"
            )

        if stage == "name":
            if len(message) < 2 or any(char.isdigit() for char in message):
                return "Please enter your name using letters, for example, Rahul Sharma."
            state["full_name"] = message
            state["stage"] = "phone"
            return "Thank you. Please enter your contact phone number."

        if stage == "phone":
            digits = re.sub(r"\D", "", message)
            if not 7 <= len(digits) <= 15:
                return "Please enter a valid phone number containing 7 to 15 digits."
            state["phone"] = digits
            state["stage"] = "date"
            return (
                "What date would you like? You can enter YYYY-MM-DD, "
                "DD-MM-YYYY, today, or tomorrow."
            )

        if stage == "date":
            parsed_date = _parse_date(message)
            if not parsed_date:
                return "I couldn't understand that date. Please enter it as YYYY-MM-DD, for example 2026-10-29."
            if date.fromisoformat(parsed_date) < date.today():
                return "That date is in the past. Please choose today or a future date."
            state["appointment_date"] = parsed_date
            state["stage"] = "time"
            return "What time would you prefer? For example, 2 or 14"

        if stage == "time":
            parsed_time = _parse_time(message)
            if not parsed_time:
                return "I couldn't understand that time. Please enter a time such as 2:00 PM or 14:00."

            state["appointment_time"] = parsed_time
            try:
                booking = book_appointment(
                    BookingRequest(
                        full_name=state["full_name"],
                        phone=state["phone"],
                        service_id=state["service_id"],
                        appointment_date=state["appointment_date"],
                        appointment_time=state["appointment_time"],
                        branch_id=1,
                        booking_source=state["mode"],
                    )
                )
            except HTTPException as exc:
                detail = exc.detail
                if exc.status_code == 409:
                    return (
                        f"{detail} Please tell me another time for "
                        f"{state['appointment_date']}."
                    )
                if exc.status_code in (400, 404):
                    return f"I couldn't create that booking: {detail} Please provide a different date or time."
                return f"Sorry, I couldn't create the booking: {detail}"

            state["booking_reference"] = booking["booking_reference"]
            state["stage"] = "confirm"
            return (
                "Please review your appointment details:\n"
                f"Name: {booking['customer_name']}\n"
                f"Service: {booking['service']}\n"
                f"Date: {booking['date']}\n"
                f"Time: {booking['time']}\n"
                f"Total: ₹{booking['total_amount']}\n"
                f"Booking reference: {booking['booking_reference']}\n"
                "Status: pending confirmation.\n\n"
                "Would you like me to confirm this appointment? Please say yes or no."
            )

        if stage == "confirm":
            if _is_yes(message):
                try:
                    result = confirm_appointment(state["booking_reference"])
                except HTTPException as exc:
                    return f"I couldn't confirm this appointment: {exc.detail}"

                reference = state["booking_reference"]
                _sessions.pop(key, None)
                return (
                    f"{result['message']} Your appointment is confirmed!\n"
                    f"Booking reference: {reference}\n"
                    "Thank you for choosing our salon."
                )

            if _is_no(message):
                reference = state["booking_reference"]
                _sessions.pop(key, None)
                return (
                    f"Okay, I haven't confirmed this appointment. "
                    f"Reference: {reference}. It may remain pending in the database; "
                    "please contact the salon or use the cancellation endpoint if you want it cancelled."
                )

            return "Would you like me to confirm the appointment? Please reply yes or no."

        _sessions.pop(key, None)
        return "Let's start again. Say 'book appointment' whenever you're ready."
