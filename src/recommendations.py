"""
Rule-based business recommendation engine mapped to
customer segments and churn risk levels.
"""

SEGMENT_RECOMMENDATIONS = {
    "Champions": [
        "Offer premium membership / VIP loyalty tier",
        "Provide early access to new product launches",
        "Invite to exclusive brand events or beta programs",
        "Request referrals and testimonials",
    ],
    "Loyal Customers": [
        "Send personalized product offers based on purchase history",
        "Enroll in a referral rewards program",
        "Provide free shipping or loyalty points multipliers",
    ],
    "Potential Loyalists": [
        "Send targeted discount coupons to encourage repeat purchase",
        "Run email/SMS engagement campaigns",
        "Recommend complementary products (cross-sell)",
    ],
    "New Customers": [
        "Send a welcome series with onboarding offers",
        "Encourage a second purchase with a limited-time discount",
        "Collect feedback via a short survey",
    ],
    "Promising Customers": [
        "Offer bundle deals to increase order value",
        "Nudge with personalized recommendations",
    ],
    "Need Attention": [
        "Send re-engagement emails highlighting new arrivals",
        "Offer limited-time discounts to reactivate interest",
    ],
    "At Risk": [
        "Launch a win-back campaign with special discounts",
        "Reach out via personalized customer service check-in",
        "Offer loyalty point bonuses for next purchase",
    ],
    "Lost Customers": [
        "Send aggressive reactivation offers (e.g., 30-40% off)",
        "Run a 'we miss you' email/SMS campaign",
        "Survey to understand reasons for disengagement",
    ],
}

RISK_RECOMMENDATIONS = {
    "High Risk": "Immediate intervention recommended: personalized discount + outreach call.",
    "Medium Risk": "Monitor closely; send a targeted engagement offer within 2 weeks.",
    "Low Risk": "No immediate action needed; continue standard engagement.",
}


def get_segment_recommendations(segment: str) -> list:
    return SEGMENT_RECOMMENDATIONS.get(segment, ["No specific recommendations available."])


def get_risk_recommendation(risk: str) -> str:
    return RISK_RECOMMENDATIONS.get(risk, "")
