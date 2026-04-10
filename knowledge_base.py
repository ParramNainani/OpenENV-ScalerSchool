from typing import Dict, Optional

KNOWLEDGE_BASE = [
    {
        "id": "POL-001",
        "title": "Refund & Return Policy",
        "content": "Refunds are available within 30 days of delivery for unused items in original packaging. We do NOT issue refunds for items that don't fit unless returned first. Original shipping fees are non-refundable. For unshipped orders, full refunds can be issued immediately upon cancellation."
    },
    {
        "id": "POL-002",
        "title": "Defective & Wrong Items Policy",
        "content": "If a customer receives a defective, damaged, or incorrect item within 90 days, offer a REPLACEMENT, not a refund. Provide a prepaid return label for the wrong/defective item. The replacement ships within 2 business days free of charge. Do NOT refund entire multi-item orders if only one item is defective."
    },
    {
        "id": "POL-003",
        "title": "Shipping & Delivery Policy",
        "content": "Standard shipping takes 5-7 business days. If an order is still within the expected delivery window, provide the tracking number and ask the customer to wait. Do NOT refund or replace items that are currently in transit and on time."
    },
    {
        "id": "POL-004",
        "title": "Escalation Policy",
        "content": "Agents should attempt to resolve issues independently. Escalate to a supervisor ONLY if: 1) The customer explicitly demands a manager/supervisor, or 2) The issue involves legal threats or safety concerns."
    }
]

PRODUCT_CATALOG = {
    "SKU-1001": {"name": "Wireless Bluetooth Headphones", "price": 79.99},
    "SKU-1002": {"name": "Men's Running Shoes (Size 10)", "price": 120.00},
    "SKU-1002-W": {"name": "Women's Running Shoes (Size 8)", "price": 120.00},
    "SKU-1003": {"name": "HD Web Camera 1080p", "price": 45.50},
    "SKU-1004": {"name": "Noise-Canceling Earbuds", "price": 199.99},
    "SKU-1005": {"name": "Ergonomic Office Chair", "price": 249.00},
    "SKU-1006": {"name": "Smart Watch Series 5", "price": 299.99},
}

ORDER_DATABASE = {
    "ORD-50123": {
        "order_id": "ORD-50123",
        "customer_name": "Sarah Mitchell",
        "items": [
            {"product_id": "SKU-1001", "product_name": "Wireless Bluetooth Headphones", "quantity": 1, "price": 79.99}
        ],
        "total": 79.99,
        "status": "shipped",
        "tracking_number": "TRK-998877",
        "days_since_order": 3,
        "delivery_expected_days": 5
    },
    "ORD-50999": {
        "order_id": "ORD-50999",
        "customer_name": "James Rodriguez",
        "items": [
            {"product_id": "SKU-1002", "product_name": "Men's Running Shoes (Size 10)", "quantity": 1, "price": 120.00}
        ],
        "total": 120.00,
        "status": "delivered",
        "tracking_number": "TRK-112233",
        "days_since_order": 6,
        "delivery_expected_days": 0
    },
    "ORD-51234": {
        "order_id": "ORD-51234",
        "customer_name": "Maria Gonzalez",
        "items": [
            {"product_id": "SKU-1003", "product_name": "HD Web Camera 1080p", "quantity": 1, "price": 45.50}
        ],
        "total": 45.50,
        "status": "processing",
        "tracking_number": None,
        "days_since_order": 1,
        "delivery_expected_days": 6
    },
    "ORD-60001": {
        "order_id": "ORD-60001",
        "customer_name": "David Chen",
        "items": [
            {"product_id": "SKU-1004", "product_name": "Noise-Canceling Earbuds", "quantity": 1, "price": 199.99},
            {"product_id": "SKU-1006", "product_name": "Smart Watch Series 5", "quantity": 1, "price": 299.99}
        ],
        "total": 499.98,
        "status": "delivered",
        "tracking_number": "TRK-445566",
        "days_since_order": 12,
        "delivery_expected_days": 0
    },
    "ORD-60002": {
        "order_id": "ORD-60002",
        "customer_name": "Emily Carter",
        "items": [
            {"product_id": "SKU-1005", "product_name": "Ergonomic Office Chair", "quantity": 1, "price": 249.00}
        ],
        "total": 249.00,
        "status": "delivered",
        "tracking_number": "TRK-778899",
        "days_since_order": 4,
        "delivery_expected_days": 0
    },
    "ORD-60003": {
        "order_id": "ORD-60003",
        "customer_name": "Alex Johnson",
        "items": [
            {"product_id": "SKU-1002", "product_name": "Men's Running Shoes (Size 10)", "quantity": 1, "price": 120.00}
        ],
        "total": 120.00,
        "status": "delivered",
        "tracking_number": "TRK-990011",
        "days_since_order": 2,
        "delivery_expected_days": 0
    }
}

def search_knowledge_base(query: str) -> list[str]:
    query = query.lower()
    keywords = [kw for kw in query.split() if len(kw) > 3]
    results = []
    
    for policy in KNOWLEDGE_BASE:
        text = f"{policy['title']} {policy['content']}".lower()
        if any(kw in text for kw in keywords) or not keywords:
            results.append(f"[{policy['title']}]: {policy['content']}")

    if not results:
        return [f"[{pol['title']}]: {pol['content']}" for pol in KNOWLEDGE_BASE]

    return results

def lookup_order(order_id: str) -> Optional[dict]:
    return ORDER_DATABASE.get(order_id)
