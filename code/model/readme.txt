To run the flask application, please execute the following lines: 

python prediction_model.py
python app.py

After this, you should see something like this
--------------------------------------------------------------------------------------------------------------------------
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000   <----------------------------------url generated for application
Press CTRL+C to quit
127.0.0.1 - - [01/Dec/2022 23:01:32] "POST /predict HTTP/1.1" 200 -
--------------------------------------------------------------------------------------------------------------------------

After this, a link will open for the app, where the user can input details and get predictions for trips.
The file prediction_model.py should be aboe to load the csv datasets present for usage.
