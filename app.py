from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
from flask_cors import CORS, cross_origin
# import flask_monitoring.dashboard as dashboard

'''Load pickel file'''
file = os.listdir('./bestmodel/')[0]
model = pickle.load(open('./bestmodel/'+file, 'rb'))
scaler = pickle.load(open('standard_scaler.pkl','rb'))

app = Flask(__name__)
# dashboard.bind(app)
# CORS(app)

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        Item_Identifier = request.form['Item ID']
        Item_Weight = float(request.form['Weight'])
        item_fat_content = request.form['Item Fat Content']

        if (item_fat_content == 'low'):
            item_fat_content = 0, 0
        elif (item_fat_content == 'reg'):
            item_fat_content = 0, 1
        else:
            item_fat_content = 1, 0

        Item_Fat_Content_1, Item_Fat_Content_2 = item_fat_content

        Item_Visibility = float(request.form['Range 0.1-2.0'])

        Item_MRP = float(request.form['Item MRP'])

        Outlet_Identifier = request.form['Outlet ID']

        Outlet_ID = Outlet_Identifier
        if (Outlet_Identifier == 'OUT013'):
            Outlet_Identifier = 1, 0, 0, 0, 0, 0, 0, 0, 0
        elif (Outlet_Identifier == 'OUT017'):
            Outlet_Identifier = 0, 1, 0, 0, 0, 0, 0, 0, 0
        elif (Outlet_Identifier == 'OUT018'):
            Outlet_Identifier = 0, 0, 1, 0, 0, 0, 0, 0, 0
        elif (Outlet_Identifier == 'OUT019'):
            Outlet_Identifier = 0, 0, 0, 1, 0, 0, 0, 0, 0
        elif (Outlet_Identifier == 'OUT027'):
            Outlet_Identifier = 0, 0, 0, 0, 1, 0, 0, 0, 0
        elif (Outlet_Identifier == 'OUT035'):
            Outlet_Identifier = 0, 0, 0, 0, 0, 1, 0, 0, 0
        elif (Outlet_Identifier == 'OUT045'):
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 1, 0, 0
        elif (Outlet_Identifier == 'OUT046'):
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 0, 1, 0
        elif (Outlet_Identifier == 'OUT049'):
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 0, 0, 1
        else:
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 0, 0, 0

        Outlet_1, Outlet_2, Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8, Outlet_9 = Outlet_Identifier

        Outlet_Year = int(2021 - int(request.form['Year']))

        Outlet_Size = request.form['Size']
        if (Outlet_Size == 'Medium'):
            Outlet_Size = 1, 0
        elif (Outlet_Size == 'Small'):
            Outlet_Size = 0, 1
        else:
            Outlet_Size = 0, 0

        Outlet_Size_1, Outlet_Size_2 = Outlet_Size

        Outlet_Location_Type = request.form['Location Type']
        if (Outlet_Location_Type == 'Tier 2'):
            Outlet_Location_Type = 1, 0
        elif (Outlet_Location_Type == 'Tier 3'):
            Outlet_Location_Type = 0, 1
        else:
            Outlet_Location_Type = 0, 0

        Outlet_Location_Type_1, Outlet_Location_Type_2 = Outlet_Location_Type

        Outlet_Type = request.form['Outlet Type']
        if (Outlet_Type == 'Supermarket Type1'):
            Outlet_Type = 1, 0, 0
        elif (Outlet_Type == 'Supermarket Type2'):
            Outlet_Type = 0, 1, 0
        elif (Outlet_Type == 'Supermarket Type3'):
            Outlet_Type = 0, 0, 1
        else:
            Outlet_Type = 0, 0, 0

        Outlet_Type_1, Outlet_Type_2, Outlet_Type_3 = Outlet_Type

        Item_Type_Combined = request.form['Item Type']

        if (Item_Type_Combined == "Food"):
            Item_Type_Combined = 1, 0
        elif (Item_Type_Combined == "Non-consumable"):
            Item_Type_Combined = 0, 1
        else:
            Item_Type_Combined = 0, 0

        Item_Type_Combined_1, Item_Type_Combined_2 = Item_Type_Combined

        data = [Item_Weight, Item_Visibility, Item_MRP, Outlet_Year, Item_Fat_Content_1, Item_Fat_Content_2,
                Outlet_Location_Type_1, Outlet_Location_Type_2, Outlet_Size_1, Outlet_Size_2, Outlet_Type_1, Outlet_Type_2,
                Outlet_Type_3, Item_Type_Combined_1, Item_Type_Combined_2, Outlet_1, Outlet_2, Outlet_3, Outlet_4, Outlet_5,
                Outlet_6, Outlet_7, Outlet_8, Outlet_9]
        features_value = [np.array(data)]

        features_name = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years',
                         'Item_Fat_Content_1', 'Item_Fat_Content_2', 'Outlet_Location_Type_1',
                         'Outlet_Location_Type_2', 'Outlet_Size_1', 'Outlet_Size_2',
                         'Outlet_Type_1', 'Outlet_Type_2', 'Outlet_Type_3',
                         'Item_Type_Combined_1', 'Item_Type_Combined_2', 'Outlet_1', 'Outlet_2',
                         'Outlet_3', 'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7', 'Outlet_8',
                         'Outlet_9']

        df = pd.DataFrame(features_value, columns=features_name)

        std_data=scaler.transform(df)

        myprd = model.predict(std_data)
        output = round(myprd[0], 2)

        # if output < 0:
        #     return render_template('index.html',prediction_texts=f"Sorry you cannot sell. Sale is negative: {output}")
        # else:
            # return render_template('result.html', prediction_text=f'The Sales production of {Item_Identifier} '
            #                                        f'by '{Outlet_ID} Outlet is  Rs {output}/-')

            # return render_template('index.html',prediction_text="Item_Outlet_Sales at {}".format(output))

        return render_template('result.html',
                                   prediction=output,
                                   Item_Identifier=Item_Identifier,
                                   Outlet_Identifier=Outlet_ID)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # port = int(os.getenv("PORT"))
    # host = '0.0.0.0'
    # httpd = simple_server.make_server(host=host, port=port, app=app)
    # httpd.serve_forever()

    app.run(debug=True)
