from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import datetime

app = Flask(__name__)

# Load model and class labels
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    disease_classes = data['classes']

# Disease to medicine mapping
disease_to_medicine = {
    "Fungal infection": ["Clotrimazole", "Fluconazole", "Ketoconazole"],
    "Allergy": ["Cetirizine", "Loratadine", "Fexofenadine"],
    "GERD": ["Omeprazole", "Ranitidine", "Esomeprazole"],
    "Chronic cholestasis": ["Ursodiol", "Cholestyramine"],
    "Drug Reaction": ["Diphenhydramine", "Prednisone", "Epinephrine (in severe cases)"],
    "Peptic ulcer diseae": ["Pantoprazole", "Amoxicillin", "Clarithromycin"],
    "AIDS": ["Zidovudine", "Lamivudine", "Efavirenz"],
    "Diabetes": ["Metformin", "Insulin", "Glipizide"],
    "Gastroenteritis": ["Oral Rehydration Salts", "Loperamide", "Zinc sulfate"],
    "Bronchial Asthma": ["Albuterol", "Fluticasone", "Montelukast"],
    "Hypertension": ["Amlodipine", "Losartan", "Hydrochlorothiazide"],
    "Migraine": ["Sumatriptan", "Ibuprofen", "Propranolol"],
    "Cervical spondylosis": ["Ibuprofen", "Cyclobenzaprine", "Physiotherapy"],
    "Paralysis (brain hemorrhage)": ["Rehabilitation Therapy", "Aspirin (if ischemic)", "Atorvastatin"],
    "Jaundice": ["Lactulose", "Cholestyramine", "Rest and hydration"],
    "Malaria": ["Artemether-lumefantrine", "Chloroquine", "Primaquine"],
    "Chicken pox": ["Acyclovir", "Calamine lotion", "Paracetamol"],
    "Dengue": ["Paracetamol", "ORS", "Platelet transfusion if needed"],
    "Typhoid": ["Cefixime", "Azithromycin", "Ciprofloxacin"],
    "hepatitis A": ["Rest", "ORS", "Vitamin supplements"],
    "Hepatitis B": ["Entecavir", "Tenofovir", "Interferon alfa-2b"],
    "Hepatitis C": ["Sofosbuvir", "Velpatasvir"],
    "Hepatitis D": ["Pegylated interferon alfa", "Supportive care"],
    "Hepatitis E": ["Rest", "Hydration", "Avoid hepatotoxic drugs"],
    "Alcoholic hepatitis": ["Prednisolone", "Pentoxifylline", "Abstinence from alcohol"],
    "Tuberculosis": ["Isoniazid", "Rifampicin", "Pyrazinamide", "Ethambutol"],
    "Common Cold": ["Paracetamol", "Antihistamines", "Decongestants"],
    "Pneumonia": ["Amoxicillin-clavulanate", "Azithromycin", "Ceftriaxone"],
    "Dimorphic hemmorhoids(piles)": ["Diosmin", "Lidocaine cream", "Hydrocortisone suppositories"],
    "Heart attack": ["Aspirin", "Clopidogrel", "Nitroglycerin", "Atorvastatin"],
    "Varicose veins": ["Compression stockings", "Diosmin", "Surgical intervention (if severe)"],
    "Hypothyroidism": ["Levothyroxine"],
    "Hyperthyroidism": ["Methimazole", "Propylthiouracil", "Beta-blockers"],
    "Hypoglycemia": ["Glucose tablets", "Glucagon injection", "Juice/sugar"],
    "Osteoarthristis": ["Paracetamol", "Diclofenac", "Glucosamine"],
    "Arthritis": ["Ibuprofen", "Methotrexate", "Sulfasalazine"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Betahistine", "Meclizine", "Epley maneuver"],
    "Acne": ["Benzoyl peroxide", "Clindamycin", "Isotretinoin"],
    "Urinary tract infection": ["Nitrofurantoin", "Ciprofloxacin", "Trimethoprim-sulfamethoxazole"],
    "Psoriasis": ["Topical corticosteroids", "Methotrexate", "Phototherapy"],
    "Impetigo": ["Mupirocin ointment", "Cephalexin", "Clindamycin"]
}

# Symptom list (keep as-is, already present)
symptom_list = [
     'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
    'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches',
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'

]

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    selected_symptoms = request.form.getlist('symptoms')

    input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    prediction_index = model.predict([input_data])[0]
    predicted_disease = disease_classes[prediction_index]
    suggested_medicines = disease_to_medicine.get(predicted_disease, ["Consult a doctor for proper medication."])

    return render_template('result.html',
                           name=name,
                           age=age,
                           gender=gender,
                           selected_symptoms=selected_symptoms,
                           predicted_disease=predicted_disease,
                           suggested_medicines=suggested_medicines)

@app.route('/download', methods=['POST'])
def download_pdf():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    selected_symptoms = request.form.getlist('symptoms')
    predicted_disease = request.form.get('predicted_disease')
    suggested_medicines = request.form.getlist('suggested_medicines')

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=50, bottomMargin=30)
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(name='HeaderStyle', fontSize=18, textColor=colors.white,
                                  alignment=1, backColor=colors.HexColor('#0f8b8d'), leading=22)
    subheader = ParagraphStyle(name='SubHeader', fontSize=13, textColor=colors.HexColor('#0f8b8d'),
                               spaceAfter=10, fontName='Helvetica-Bold')
    label_style = ParagraphStyle(name='Label', fontSize=11, textColor=colors.black, spaceAfter=6)
    value_style = ParagraphStyle(name='Value', fontSize=11, textColor=colors.darkgray, spaceAfter=12)

    story = []
    story.append(Paragraph("🩺 DISEASENET - PREDICTION REPORT", header_style))
    story.append(Spacer(1, 14))
    story.append(Paragraph("📋 Patient Information", subheader))
    story.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", label_style))
    story.append(Paragraph(f"<b>Name:</b> {name}", label_style))
    story.append(Paragraph(f"<b>Age:</b> {age}", label_style))
    story.append(Paragraph(f"<b>Gender:</b> {gender}", label_style))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 14))

    story.append(Paragraph("🧾 Selected Symptoms", subheader))
    if selected_symptoms:
        symptoms_table = [[symptom.replace('_', ' ').capitalize()] for symptom in selected_symptoms]
    else:
        symptoms_table = [["None"]]

    table = Table(symptoms_table, colWidths=[460])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightcyan])
    ]))
    story.append(table)

    story.append(Spacer(1, 20))
    story.append(Paragraph("🔍 Prediction Result", subheader))
    story.append(Paragraph(f"<b>Predicted Disease:</b> <font color='red'>{predicted_disease}</font>", value_style))

    story.append(Spacer(1, 14))
    story.append(Paragraph("💊 Recommended Medicines", subheader))
    for med in suggested_medicines:
        story.append(Paragraph(f"- {med}", value_style))

    story.append(Spacer(1, 30))
    story.append(Paragraph("🤖 This report was generated by DiseaseNet AI-based Disease Prediction System.",
                           ParagraphStyle('Footer', fontSize=9, textColor=colors.grey, alignment=1)))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name=f"{name.replace(' ', '_')}_Disease_Report.pdf",
                     mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)