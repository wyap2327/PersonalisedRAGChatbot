from faker import Faker
import json
import random

fake = Faker()

def generate_customer_record():
    return {
        "customer_id": fake.uuid4(),
        "name": fake.name(),
        "email": fake.email(),
        "age": random.randint(18, 65),
        "membership_tier": random.choice(["Bronze", "Silver", "Gold"]),
        "phone": fake.phone_number(),
        "address": fake.address(),
        "join_date": fake.date_between(start_date="-3y", end_date="today").isoformat(),
    }

def generate_order(customer_id):
    return {
        "order_id": fake.uuid4(),
        "customer_id": customer_id,
        "product": fake.catch_phrase(),
        "status": random.choice(["delivered", "shipped", "processing", "cancelled"]),
        "order_date": fake.date_between(start_date="-1y", end_date="today").isoformat(),
        "total": round(random.uniform(10.0, 500.0), 2),
    }

# Generate 15 customers with order history
customers = []
for _ in range(15):
    customer = generate_customer_record()
    customer["orders"] = [generate_order(customer["customer_id"]) for _ in range(random.randint(1, 5))]
    customers.append(customer)

# Save to JSON
with open("customer_records.json", "w") as f:
    json.dump(customers, f, indent=2)

print(f"Generated {len(customers)} customer records")