import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.python.keras.saving.save import load_model

app = Flask(__name__)

# Load model
loaded_model = load_model('model.h5')
print("Model diload")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('pages/input/form.html')

@app.route('/about')
def about():
    return render_template('pages/about/about.html')

@app.route('/predict',methods=['POST'])
def predict():

    SKS = request.form['SKS']
    SKS = float(SKS)

    Semester = request.form['Semester']
    Semester = float(Semester)

    IPK = request.form['IPK']
    IPK = float(IPK)

    tes = [[SKS, Semester, IPK]]
    
    prediction = loaded_model.predict_classes(tes)
    
    if prediction == 1:
        prediksi = ("berpeluang lulus jika terus melanjutkan")
    else :
        prediksi = ("memiliki peluang kecil untuk lulus, sebaiknya di pertimbangkan")

    return render_template('pages/input/form.html', prediction_text='Prediksi model menunjukkan anda {}'.format(prediksi))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = loaded_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)