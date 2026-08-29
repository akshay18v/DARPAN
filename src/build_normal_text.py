import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "normal_text_dataset.csv"
)

normal_texts = [

    # Search / navigation
    "Search hotels",
    "Search flights",
    "Search",
    "Home",
    "Hotels",
    "Flights",
    "Holidays",
    "Tours",
    "Cars",
    "Trains",
    "Buses",
    "Customer support",
    "Help",
    "Contact us",
    "Login",
    "Sign up",

    # Dates / guests
    "Check-in",
    "Check-out",
    "Select check-in date",
    "Select check-out date",
    "Choose date",
    "Select date",
    "1 room, 2 guests",
    "2 adults",
    "1 traveller",
    "Travellers and cabin class",

    # Search forms
    "From",
    "To",
    "Departure",
    "Arrival",
    "Destination",
    "Enter destination",
    "Select destination",
    "Modify search",
    "Search hotels",
    "Search flights",

    # Normal hotel UI
    "View details",
    "View all rooms",
    "Choose your room",
    "Room details",
    "Room amenities",
    "Hotel amenities",
    "Property details",
    "Photos",
    "Reviews",
    "Ratings and reviews",
    "Location",
    "Facilities and services",
    "What's nearby",
    "Cancellation policy",

    # Normal flight UI
    "Flight details",
    "View fares",
    "Select flight",
    "Fare options",
    "Departure time",
    "Arrival time",
    "Flight duration",
    "Direct flight",
    "Non-stop",
    "Baggage",
    "Checked baggage",
    "Hand baggage",
    "Meal",
    "Seat selection",

    # Normal booking UI
    "Enter your details",
    "Full name",
    "Email address",
    "Mobile number",
    "Payment details",
    "Payment methods",
    "Payment summary",
    "Booking details",
    "Trip details",
    "Booking confirmation",
    "Booking reference",
    "Total amount",
    "Price details",
    "Fare summary",
    "Terms and conditions",
    "Privacy policy",

    # Normal buttons
    "Continue",
    "Continue to booking",
    "Continue to checkout",
    "Proceed",
    "Proceed to payment",
    "Book now",
    "Confirm booking",
    "Pay now",
    "Submit",
    "Apply",
    "Clear all",
    "Show more",
    "Show less",
    "Back",
    "Close",

    # Normal filters
    "Filters",
    "Sort by",
    "Price",
    "Rating",
    "Review score",
    "Amenities",
    "Hotel class",
    "Meal plan",
    "Stops",
    "Airlines",
    "Free cancellation",

    # Normal informational text
    "Breakfast included",
    "Free Wi-Fi",
    "Parking available",
    "Private bathroom",
    "Air conditioning",
    "Swimming pool",
    "Restaurant",
    "Family room",
    "Double room",
    "Twin room",
    "Standard room",
    "Deluxe room",
    "1 night",
    "2 nights",
    "Per night",
    "2 adults and 1 child",
]


# ------------------------------------------------------------
# Create rows
# ------------------------------------------------------------

rows = []

for text in normal_texts:

    rows.append({
        "text": text,
        "label": "normal",
        "source": "manual_normal",
        "image_name": "",
        "website": "generic",
        "pattern_type": "normal",
        "comment": "Normal UI text - no dark pattern"
    })


df = pd.DataFrame(rows)

# Remove duplicates
df = df.drop_duplicates(
    subset=["text", "label"]
).reset_index(drop=True)

# Save
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("DARPAN - NORMAL UI TEXT DATASET")
print("=" * 70)

print("\nNormal samples:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nColumns:")
print(list(df.columns))

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDone.")