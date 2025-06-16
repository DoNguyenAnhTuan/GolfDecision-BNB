```markdown
# 🏌️ Golf Decision using Naive Bayes Classifier (BNB)

This repository presents a simple implementation of the **Naive Bayes Classifier** in Java for the classic **golf decision dataset**, where the model predicts whether to play golf based on weather conditions such as outlook, temperature, humidity, and wind.

---

## 🧠 Algorithm

- **Naive Bayes** (Bernoulli variant)
- Probabilistic classifier based on Bayes' Theorem with independence assumptions
- Handles categorical input features efficiently

---

## 🗃️ Dataset Structure

| Outlook  | Temperature | Humidity | Wind   | PlayGolf |
|----------|-------------|----------|--------|----------|
| Sunny    | Hot         | High     | Weak   | No       |
| Overcast | Mild        | Normal   | Strong | Yes      |
| Rainy    | Cool        | High     | Weak   | No       |
| ...      | ...         | ...      | ...    | ...      |

---

## 📂 Project Structure

```

GolfDecision-BNB/
├── src/
│   └── Main.java
│   └── NaiveBayesClassifier.java
│   └── Dataset.java
├── dataset.csv
└── README.md

````

---

## 🚀 How to Run

1. Compile the program:
```bash
javac -d bin src/*.java
````

2. Run the classifier:

```bash
java -cp bin Main
```

The classifier will train on the dataset and output prediction results for new examples.

---

## 📈 Sample Output

```
Training accuracy: 92.3%
Predicting new sample: {Sunny, Cool, High, Strong}
→ Result: No
```

---

## 📚 Learning Outcomes

* Understand how Naive Bayes works on categorical datasets
* Gain experience in implementing probabilistic models in Java
* See real application on decision-based problems

---

## 👨‍💻 Author

**Do Nguyen Anh Tuan**
🎓 MSc Student in IT @ Lac Hong University
🏢 FabLab @ EIU
🔗 [Portfolio Website](https://donguyenanhtuan.github.io/AnhTuan-Portfolio)


