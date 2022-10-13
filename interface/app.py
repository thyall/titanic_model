import gradio as gr
import joblib as jb


#function for prediction model
def predict(sex, age, pclass):
    model = jb.load("titanic_modelo.pkl")
    pclass = int(pclass)
    p = model.predict_proba([[sex, age, pclass]])[0]

    return { "Didn't survive": p[0], "Survived": p[1]}


#interface
demo = gr.Interface(fn=predict,
    inputs=[gr.Dropdown(choices=["Male", "Famale"], type="index"),
    "number",
    gr.Dropdown(choices=["1","2","3"], type="value")], outputs="label")

demo.launch()