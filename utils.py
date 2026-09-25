"""
=============================================================
  SleepIQ — Shared utility functions
  Used by app.py (recommendations + risk score).
=============================================================
"""


def get_recommendations(stress, sleep_duration, bmi_category, physical_activity, disorder):
    """
    Returns (disorder_name: str, recommendations: list[str]).
    disorder: int  0=Healthy | 1=Insomnia | 2=Sleep Apnea
    """
    class_names = {0: 'Healthy', 1: 'Insomnia', 2: 'Sleep Apnea'}
    disorder_name = class_names.get(int(disorder), 'Unknown')
    recs = []

    # Stress
    if stress >= 7:
        recs += [
            "😰 HIGH STRESS: Practice deep breathing or meditation daily.",
            "📵 Limit screen time 1 hour before bed.",
            "📓 Try journaling to offload thoughts before sleep.",
        ]
    elif stress >= 5:
        recs += [
            "😐 MODERATE STRESS: Light yoga or stretching before bed.",
            "🕐 Maintain a consistent sleep schedule.",
        ]
    else:
        recs.append("✅ Good stress management — keep it up!")

    # Sleep duration
    if sleep_duration < 6:
        recs += [
            "😴 INSUFFICIENT SLEEP: Aim for 7–9 hours per night.",
            "☕ Avoid caffeine after 3 PM.",
        ]
    elif sleep_duration > 9:
        recs.append("💤 EXCESSIVE SLEEP may signal a disorder — consult a specialist.")
    else:
        recs.append("✅ Sleep duration is in the healthy range (7–9 hrs).")

    # BMI
    if bmi_category == 2:
        recs += [
            "⚖️ OBESE BMI: Weight management may reduce Sleep Apnea risk.",
            "🥗 Consult a nutritionist for a personalised plan.",
        ]
    elif bmi_category == 1:
        recs.append("⚖️ OVERWEIGHT: Regular cardio exercise is recommended.")
    else:
        recs.append("✅ BMI is in the normal range.")

    # Physical activity
    if physical_activity < 30:
        recs += [
            "🏃 LOW ACTIVITY: Aim for 30 min of moderate exercise daily.",
            "🌙 Avoid intense workouts within 2 hours of bedtime.",
        ]
    elif physical_activity >= 60:
        recs.append("✅ Great physical activity level — well done!")
    else:
        recs.append("🏃 Moderate activity — try to build up gradually.")

    # Disorder-specific
    if disorder == 1:
        recs += [
            "🌙 INSOMNIA: Consider Cognitive Behavioural Therapy (CBT-I).",
            "❄️ Keep bedroom cool, dark, and quiet.",
            "🚫 Avoid naps longer than 20 minutes.",
        ]
    elif disorder == 2:
        recs += [
            "😤 SLEEP APNEA: Consult a doctor about CPAP therapy.",
            "🛌 Sleep on your side instead of your back.",
            "🚫 Avoid alcohol and sedatives before sleep.",
        ]
    else:
        recs.append("🎉 HEALTHY: Maintain your current lifestyle habits!")

    return disorder_name, recs


def sleep_risk_score(stress, sleep_duration, bmi_category, physical_activity):
    """
    Calculate a 0–100 sleep risk score.
    Returns: (score: float, risk_level: str)
    """
    score = 0
    score += min(stress * 3.75, 30)
    score += 25 if sleep_duration < 6 else (15 if sleep_duration < 7 else 0)
    score += bmi_category * 12.5
    score += max(0, (60 - physical_activity) / 60 * 20)
    score = min(score, 100)

    if score < 30:
        risk_level = "🟢 Low Risk"
    elif score < 60:
        risk_level = "🟡 Moderate Risk"
    else:
        risk_level = "🔴 High Risk"

    return round(score, 1), risk_level
