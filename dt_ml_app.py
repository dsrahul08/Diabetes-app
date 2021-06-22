import streamlit as st 
import joblib
import os
import numpy as np


target_label_map = {"Negative":0,"Positive":1}

['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
       'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
       'itching', 'irritability', 'delayed_healing', 'partial_paresis',
       'muscle_stiffness', 'alopecia', 'obesity', 'class']


# Load ML Models
@st.cache
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def run_dt_app():
	st.subheader("Machine Learning Section")
	st.image("images/dtplot.png")
	loaded_model = load_model("models/decision_tree_model_diabetes_22_06_2020.pkl")
    
	# Layout
	col1,col2 = st.beta_columns(2)

	with col1:
		age = st.number_input("Age",10,100)
		TricepsThickness = st.number_input("TricepsThickness",10,100)
		BMI = st.number_input("BMI",10,100)
		SerumInsulin = st.number_input("SerumInsulin",10,100)
	
	with col2:
		pregnancies = st.number_input("Pregnancies",0,20)
		PlasmaGlucose = st.number_input("PlasmaGlucose",10,100)
		DiastolicBP = st.number_input("DiastolicBP",10,100)
		DiabetesFn = st.number_input("DiabetesFn",10,100)
        
	with st.beta_expander("Your Selected Options"):
		result = {'age':age,
		'pregnancies':pregnancies,
		'TricepsThickness':TricepsThickness,
		'BMI':BMI,
		'PlasmaGlucose':PlasmaGlucose,
		'DiastolicBP':DiastolicBP,
		'SerumInsulin':SerumInsulin,
		'DiabetesFn':DiabetesFn}
		st.write(result)
		encoded_result = []
		for i in result.values():
			if type(i) == int:
				encoded_result.append(i)
			elif i in ["Female","Male"]:
				res = get_value(i,gender_map)
				encoded_result.append(res)
			else:
				encoded_result.append(get_fvalue(i))

		# st.write(encoded_result)
	with st.beta_expander("Prediction Results"):
		single_sample = np.array(encoded_result).reshape(1,-1)

		
		prediction = loaded_model.predict(single_sample)
		pred_prob = loaded_model.predict_proba(single_sample)
		st.write(prediction)
		if prediction == 1:
			st.warning("Positive Risk-{}".format(prediction[0]))
			pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)
		else:
			st.success("Negative Risk-{}".format(prediction[0]))
			pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)
            
