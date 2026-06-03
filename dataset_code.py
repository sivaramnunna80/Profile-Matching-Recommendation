import pandas as pd
import random
from faker import Faker

fake = Faker()

# -----------------------------
# Sample Data Pools
# -----------------------------

mbti_types = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

skills_pool = [
    "Python", "Machine Learning", "Deep Learning", "NLP",
    "Data Analysis", "UI/UX", "Cybersecurity", "Java",
    "C++", "Cloud Computing", "SQL", "TensorFlow",
    "PyTorch", "Web Development", "React", "Computer Vision"
]

interests_pool = [
    "Photography", "Gaming", "Reading", "Traveling",
    "AI Research", "Music", "Fitness", "Startups",
    "Blogging", "Movies", "Design", "Public Speaking"
]

career_goals_pool = [
    "Become an AI Engineer",
    "Start a tech company",
    "Work in Data Science",
    "Build innovative products",
    "Research in Machine Learning",
    "Become a UI/UX expert",
    "Develop scalable software systems",
    "Work in cloud architecture"
]

work_styles = [
    "Collaborative", "Independent",
    "Leadership-focused", "Creative",
    "Analytical", "Flexible"
]

personal_values_pool = [
    "Innovation", "Honesty", "Teamwork",
    "Discipline", "Creativity", "Empathy",
    "Growth", "Curiosity"
]

locations = [
    "Hyderabad", "Bangalore", "Chennai",
    "Mumbai", "Delhi", "Pune",
    "Kolkata", "Visakhapatnam"
]

professions = [
    "Data Scientist", "Software Engineer",
    "AI Researcher", "UI/UX Designer",
    "ML Engineer", "Cloud Engineer",
    "Backend Developer", "Cybersecurity Analyst"
]

# -----------------------------
# Generate Dataset
# -----------------------------

data = []

for i in range(1, 101):

    about_me = (
        f"I am passionate about {random.choice(interests_pool)} "
        f"and enjoy working on {random.choice(skills_pool)} projects."
    )

    professional_summary = (
        f"Experienced in {random.choice(skills_pool)} and "
        f"{random.choice(skills_pool)} with strong problem-solving skills."
    )

    row = {
        "user_id": i,
        "name": fake.name(),
        "age": random.randint(21, 35),
        "gender": random.choice(["Male", "Female","Others"]),
        "location": random.choice(locations),
        "profession": random.choice(professions),
        "education": random.choice(["B.Tech", "M.Tech", "MBA", "B.Sc", "MCA"]),
        "experience_years" :random.randint(1,5),
        "mbti_type": random.choice(mbti_types),
        "about_me": about_me,
        "professional_summary": professional_summary,
        "interests": ", ".join(random.sample(interests_pool, 3)),
        "skills": ", ".join(random.sample(skills_pool, 4)),
        "career_goals": random.choice(career_goals_pool),
        "work_style": random.choice(work_styles),
        "personal_values": ", ".join(random.sample(personal_values_pool, 3)),
    }

    data.append(row)

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame(data)

# -----------------------------
# Save Dataset
# -----------------------------

df.to_csv("dataset.csv", index=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
print(df.head())