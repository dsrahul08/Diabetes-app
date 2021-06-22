import streamlit as st 
import streamlit.components.v1 as stc 
from logistic_ml_app import run_logistic_app
from dt_ml_app import run_dt_app

html_temp = """
		<div style="background-color:#03C04A;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Diabetes Prediction App</h1>
		<h4 style="color:white;text-align:center;">One of the critical diseases</h4>
		</div>
		"""
st.image("images/logo.png")
def main():
    
	# st.title("ML Web App with Streamlit")
	stc.html(html_temp)

	menu = ["Home","Logistic Regression","Decision Tree"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		st.write("""
			### Early Stage Diabetes Risk Predictor App
			Welcome to the Smart Cube App which will predict the early stage diabetes. 
			""")
	elif choice == "Logistic Regression":
		run_logistic_app()
	elif choice == "Decision Tree":
		run_dt_app()

if __name__ == '__main__':
	main()