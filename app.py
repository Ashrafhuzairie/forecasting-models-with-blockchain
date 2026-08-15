from flask import Flask, jsonify, render_template

from warehouse_forecasting.paper_results import table


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", author="Mohd Ashraf Huzairie")


@app.get("/api/paper-results")
def paper_results():
    return jsonify(table().to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=False)
