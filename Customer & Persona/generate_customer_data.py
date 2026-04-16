"""
generate_customer_data.py

Generates synthetic customer records for the ShopNest e-commerce chatbot evaluation.
Customer sport profiles are loaded from customer_profiles.json.
Personal identity fields (name, email, phone, address, DOB) are generated using Faker.

Outputs:
  - customer_data/customers.json       (full customer records for RAG pipeline lookup)
  - customer_data/personas/*.txt       (one persona file per customer for ChromaDB embedding)

Requirements: pip install faker
"""

import json
import os
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("en_GB")
random.seed(42)
Faker.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMER_DATA_DIR = os.path.join(BASE_DIR, "customer_data")
PERSONAS_DIR = os.path.join(CUSTOMER_DATA_DIR, "personas")
os.makedirs(PERSONAS_DIR, exist_ok=True)

PRODUCTS = {
    "NIKE-AM270-001":    {"name": "Nike Air Max 270",                         "price": 129.99},
    "ADID-UB22-002":    {"name": "Adidas Ultraboost 22",                      "price": 159.99},
    "NB-FFX1080-V13":   {"name": "New Balance Fresh Foam X 1080v13",          "price": 154.99},
    "ASIC-GN25-003":    {"name": "ASICS Gel-Nimbus 25",                       "price": 164.99},
    "PUMA-VN2-004":     {"name": "Puma Velocity Nitro 2",                     "price": 109.99},
    "CONV-CT-HI-005":   {"name": "Converse Chuck Taylor All Star Hi",         "price":  64.99},
    "VANS-OS-006":      {"name": "Vans Old Skool",                            "price":  74.99},
    "NIKE-AF1-007":     {"name": "Nike Air Force 1 '07",                      "price":  94.99},
    "ADID-PRED-FG-008": {"name": "Adidas Predator Accuracy.3 FG Boots",      "price":  74.99},
    "NIKE-MET9-009":    {"name": "Nike Metcon 9 Training Shoe",               "price": 124.99},
    "NIKE-DFIT-MT-010": {"name": "Nike Dri-FIT Running T-Shirt (Men's)",      "price":  34.99},
    "ADID-OTR-WLT-011": {"name": "Adidas Own the Run Long Sleeve (Women's)",  "price":  44.99},
    "UA-HG-COMP-012":   {"name": "Under Armour HeatGear Compression Top",    "price":  39.99},
    "NIKE-PE-WT-013":   {"name": "Nike Phenom Elite Running Tights (Women's)","price":  74.99},
    "ADID-SQ21-MS-014": {"name": "Adidas Squadra 21 Shorts (Men's)",          "price":  17.99},
    "GS-VS2-WL-015":    {"name": "Gymshark Vital Seamless 2.0 Leggings",      "price":  49.99},
    "TNF-WW-MJ-016":    {"name": "The North Face Windwall Running Jacket",    "price": 109.99},
    "NIKE-TFA-DT-017":  {"name": "Nike Therma-FIT Academy Drill Top",         "price":  54.99},
    "FITB-CHG6-BLK-018":{"name": "Fitbit Charge 6",                          "price": 139.99},
    "GS-LSTP-019":      {"name": "Gymshark Lifting Straps",                   "price":  14.99},
    "YNOW-YM-6MM-020":  {"name": "Yoga Mat — Non-Slip TPE 6mm",               "price":  29.99},
    "NIKE-HWSET-021":   {"name": "Nike Dri-FIT Headband and Wristband Set",   "price":  16.99},
    "ADID-STAD2-GB-022":{"name": "Adidas Stadium II Gym Bag 34L",             "price":  39.99},
}


def make_order(sku_list, days_min, days_max, status="Delivered"):
    skus = random.sample(sku_list, k=random.randint(1, min(3, len(sku_list))))
    items = [
        {"sku": s, "product_name": PRODUCTS[s]["name"], "price": PRODUCTS[s]["price"], "quantity": 1}
        for s in skus
    ]
    order_total = round(sum(i["price"] for i in items), 2)
    delta = random.randint(days_min, days_max)
    order_date = (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%d")
    return {
        "order_id": f"SN{fake.numerify('######')}",
        "date": order_date,
        "status": status,
        "items": items,
        "order_total": order_total,
        "delivery_method": random.choice(["Standard Delivery", "Express Delivery", "Next Day Delivery"]),
        "points_earned": int(order_total),
    }


def build_order_history(skus):
    num_orders = random.randint(2, 5)
    orders = []
    for i in range(num_orders):
        days_min = 30 + i * 60
        days_max = days_min + 59
        status = "Delivered" if i < num_orders - 1 else random.choice(["Delivered", "Dispatched", "Processing"])
        orders.append(make_order(skus, days_min, days_max, status))
    return sorted(orders, key=lambda o: o["date"], reverse=True)


def generate_customer(profile):
    # Faker generates all personal identity fields
    first_name = fake.first_name()
    last_name = fake.last_name()
    order_history = build_order_history(profile["order_skus"])
    total_spent = round(sum(o["order_total"] for o in order_history), 2)

    return {
        "customer_id": profile["customer_id"],
        "personal_details": {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "email": fake.email(),
            "phone": fake.phone_number(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=55).isoformat(),
            "address": {
                "line1": fake.street_address(),
                "city": fake.city(),
                "county": fake.county(),
                "postcode": fake.postcode(),
                "country": "United Kingdom",
            },
            "account_created": fake.date_between(start_date="-3y", end_date="-6m").isoformat(),
        },
        "account": {
            "membership_tier": profile["membership_tier"],
            "loyalty_points_balance": profile["loyalty_points_balance"],
            "total_lifetime_spend": total_spent,
            "total_orders": len(order_history),
        },
        "preferences": {
            "sport_interests": profile["sport_interests"],
            "preferred_brands": profile["preferred_brands"],
            "shoe_size_uk": profile["shoe_size_uk"],
            "clothing_size": profile["clothing_size"],
        },
        "order_history": order_history,
        "past_interactions": profile["past_interactions"],
        "persona_summary": profile["persona_summary"].format(first_name=first_name),
    }


def write_persona_file(customer):
    p = customer["personal_details"]
    a = customer["account"]
    pref = customer["preferences"]
    recent_products = [
        item["product_name"]
        for order in customer["order_history"][:3]
        for item in order["items"]
    ]
    lines = [
        f"CUSTOMER PROFILE — {p['full_name']} ({customer['customer_id']})",
        "=" * 60,
        f"Name: {p['full_name']}",
        f"Email: {p['email']}",
        f"Membership Tier: {a['membership_tier']}",
        f"Loyalty Points Balance: {a['loyalty_points_balance']} points",
        f"Total Lifetime Spend: £{a['total_lifetime_spend']:.2f}",
        f"Total Orders: {a['total_orders']}",
        f"Sport Interests: {', '.join(pref['sport_interests'])}",
        f"Preferred Brands: {', '.join(pref['preferred_brands'])}",
        f"Shoe Size (UK): {pref['shoe_size_uk']}",
        f"Clothing Size: {pref['clothing_size']}",
        "",
        "About This Customer:",
        customer["persona_summary"],
        "",
        "Recent Purchases:",
        *[f"  - {p}" for p in recent_products[:5]],
        "",
        "Recent Support Interactions:",
        *[f"  [{i['date']}] {i['type'].upper()}: {i['detail']}"
          for i in customer["past_interactions"]],
    ]
    filename = f"{customer['customer_id']}_{customer['personal_details']['last_name'].lower()}.txt"
    with open(os.path.join(PERSONAS_DIR, filename), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    profiles_path = os.path.join(BASE_DIR, "customer_profiles.json")
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    all_customers = []
    for profile in profiles:
        customer = generate_customer(profile)
        all_customers.append(customer)
        write_persona_file(customer)
        print(f"  Generated: {customer['personal_details']['full_name']:<25} "
              f"({customer['account']['membership_tier']} tier)")

    output_path = os.path.join(CUSTOMER_DATA_DIR, "customers.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_customers, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_customers)} customers to {output_path}")


if __name__ == "__main__":
    main()
