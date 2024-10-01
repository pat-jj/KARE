from pyhealth.data import Patient, Visit

# LUNG_CANCER_CODES = [162.2, 162.3, 162.4, 162.5, 162.8, 162.9]
# LUNG_CANCER_CODES = set([str(int(code*10)) for code in LUNG_CANCER_CODES])
LUNG_CANCER_CODES = ["19"]
    
def drug_recommendation_mimic3_fn(patient):
    samples = []
    for i in range(len(patient)):
        visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        # drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples

def drug_recommendation_mimic4_fn(patient: Patient):

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


def mortality_prediction_mimic3_fn(patient):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit = patient[i]
        next_visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": mortality_label,
            }
        )

    return samples

def mortality_prediction_mimic3_fn_lung(patient):
    samples = []
    lung_cancer_label = 0
    for i in range(len(patient)):
        visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        for code in conditions:
            if code in LUNG_CANCER_CODES:
                lung_cancer_label = 1
                break
        if lung_cancer_label == 1:
            break

    if lung_cancer_label == 1:
        for i in range(len(patient) - 1):
            visit = patient[i]
            next_visit = patient[i + 1]

            if next_visit.discharge_status not in [0, 1]:
                mortality_label = 0
            else:
                mortality_label = int(next_visit.discharge_status)

            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            # exclude: visits without condition, procedure, and drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue
            
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "label": mortality_label,
                }
            )        
    else: 
        return [{
            "visit_id": patient[0].visit_id,
            "patient_id": patient.patient_id,
            "conditions": ['None'],
            "procedures": ['None'],
            "drugs": ['None'],
            "label": 1000,
        }]

    return samples


def lung_cancer_prediction_mimic3_fn(patient: Patient):
    samples = []
    lung_cancer_label = 0
    lung_cancer_visit_index = -1

    # Identify if and when the patient develops lung cancer
    for i in range(len(patient)):
        visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        for code in conditions:
            if code in LUNG_CANCER_CODES:
                lung_cancer_label = 1
                lung_cancer_visit_index = i
                break
        if lung_cancer_label == 1:
            break

    # Process samples based on lung cancer status
    if lung_cancer_label == 1:
        # For patients who develop lung cancer, use visits before diagnosis
        for i in range(lung_cancer_visit_index):
            visit = patient[i]
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            
            
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "label": lung_cancer_label,
                }
            )
    else:
        visit_indices = list(range(len(patient) - 1))
        for i in visit_indices[:min(5, len(visit_indices))]:  # Limit to 5 samples per patient
            visit = patient[i]
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue
            
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "label": lung_cancer_label,
                }
            )

    return samples



def readmission_prediction_mimic3_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label,
            }
        )
        
    if len(samples) < 1:
        return []
    # no cohort selection
    return samples

def length_of_stay_prediction_mimic3_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
    # no cohort selection
    return samples


def length_of_stay_prediction_mimic4_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
    # no cohort selection
    return samples


def length_of_stay_prediction_eicu_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
    # no cohort selection
    return samples

def mortality_prediction_mimic4_fn(patient: Patient):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def readmission_prediction_mimic4_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9