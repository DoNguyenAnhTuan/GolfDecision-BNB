import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
import gradio as gr
import numpy as np
import seaborn as sns
# Thư viện tạo biểu đồ và hình ảnh
import matplotlib.pyplot as plt
# Hỗ trợ tải và hiển thị hình ảnh
import matplotlib.image as mpimg
# Thư viện xử lý ảnh (mở, thao tác hình ảnh)
from PIL import Image, ImageEnhance

try:
    # Read the CSV file, assuming it's in the same directory as the script
    df = pd.read_csv("playsheet_dataset.csv")
except FileNotFoundError:
    print("Error: CSV file 'playsheet_dataset.csv' not found. Please ensure it's in the same directory as this script or provide the correct path.")
    exit()

# Print the DataFrame to get a general overview
print(df)


df[df['Outlook']==min(df.Outlook)]
plt.figure(figsize=(12,6))
sns.histplot(data=df,x=df.Outlook,bins=20)
x=df.Outlook.value_counts()
plt.figure(figsize=(12, 10))
plt.pie(x, labels=x.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'gold', 'lightgray', 'lightgreen', 'lightcoral'])
plt.title('Phân bố các loại thời tiết')
plt.savefig('image/Outlook.jpg')


df[df['Temp']==min(df.Temp)]
plt.figure(figsize=(12,6))
sns.histplot(data=df,x=df.Temp,bins=20)
x=df.Temp.value_counts()
plt.figure(figsize=(12, 10))
plt.pie(x, labels=x.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'gold', 'lightgray', 'lightgreen', 'lightcoral'])
plt.title('Phân bố các loại nhiệt độ')
plt.savefig('image/Temp.jpg')


df[df['Humidity']==min(df.Humidity)]
plt.figure(figsize=(12,6))
sns.histplot(data=df,x=df.Humidity,bins=20)
x=df.Humidity.value_counts()
plt.figure(figsize=(12, 10))
plt.pie(x, labels=x.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'gold', 'lightgray', 'lightgreen', 'lightcoral'])
plt.title('Phân bố các loại độ ẩm')
plt.savefig('image/Humidity.jpg')


df[df['Windy']==min(df.Windy)]
plt.figure(figsize=(12,6))
sns.histplot(data=df,x=df.Windy,bins=20)
x=df.Windy.value_counts()
plt.figure(figsize=(12, 10))
plt.pie(x, labels=x.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'gold', 'lightgray', 'lightgreen', 'lightcoral'])
plt.title('Phân bố các loại gió')
plt.savefig('image/Windy.jpg')


# Encode categorical features using LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])

# Separate features and target variable
features = df.drop('Play', axis=1)  # All columns except 'Play'
target = df['Play']

# Split data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a single dictionary for all feature encodings (modify as needed)
encoding_map = {
  'Outlook': {'Âm u': 0, 'Mưa': 1, 'Nắng': 2},
  'Temp': {'Lạnh': 0, 'Nóng': 1, 'Ôn hòa': 2},
  'Humidity': {'Cao': 0, 'Bình thường': 1},
  'Windy': {'Không': 0, 'Có': 1},
  'Play': {"Không chơi": 0, "Chơi": 1}
}
model = BernoulliNB()
model.fit(X_train, y_train)

# Evaluate model performance on test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
confusion_matrix_ = confusion_matrix(y_test, predictions)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_matrix_)
# Calculate average precision, recall, and F1 scores across all classes (optional)
avg_precision = precision_score(y_test, predictions, average='weighted')
avg_recall = recall_score(y_test, predictions, average='weighted')
avg_f1 = f1_score(y_test, predictions, average='weighted')

# Option 1: Individual bar charts for each metric
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # Adjust figure size as needed

axs[0].bar(np.arange(len(le.classes_)), precision, color='skyblue', label='Precision')
axs[0].set_xlabel('Class')
axs[0].set_ylabel('Precision Score')
axs[0].set_xticks(np.arange(len(le.classes_)))
axs[0].set_xticklabels(le.classes_, rotation=45, ha='right')
axs[0].legend()

axs[1].bar(np.arange(len(le.classes_)), recall, color='coral', label='Recall')
axs[1].set_xlabel('Class')
axs[1].set_ylabel('Recall Score')
axs[1].set_xticks(np.arange(len(le.classes_)))
axs[1].set_xticklabels(le.classes_, rotation=45, ha='right')
axs[1].legend()

axs[2].bar(np.arange(len(le.classes_)), f1, color='gold', label='F1-Score')
axs[2].set_xlabel('Class')
axs[2].set_ylabel('F1 Score')
axs[2].set_xticks(np.arange(len(le.classes_)))
axs[2].set_xticklabels(le.classes_, rotation=45, ha='right')
axs[2].legend()

fig.suptitle('Model Performance by Class')
plt.tight_layout()
plt.savefig('image/P,R,F1.jpg')
# plt.show()
def predict_play(outlook, temp, humidity, windy):
    model = BernoulliNB()
    model.fit(X_train, y_train)

    # Evaluate model performance on test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    confusion_matrix_ = confusion_matrix(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix_)
    # Check if any input is invalid (not in the encoding_map)
    invalid_input = False
    for feature, value in zip(['Outlook', 'Temp', 'Humidity', 'Windy'], [outlook, temp, humidity, windy]):
        if value not in encoding_map[feature]:
            invalid_input = True
            return "Invalid input. Please use the specified options for each feature."

    # Encode user input using the dictionary
    outlook_encoded = encoding_map['Outlook'].get(outlook)
    temp_encoded = encoding_map['Temp'].get(temp)
    humidity_encoded = encoding_map['Humidity'].get(humidity)
    windy_encoded = encoding_map['Windy'].get(windy)

    # Create a new data point for prediction
    new_data = [[outlook_encoded, temp_encoded, humidity_encoded, windy_encoded]]
    
    # Create the Gaussian Naive Bayes classifier (if not already created)
    global clf
    if not hasattr(predict_play, 'clf'):
        clf = BernoulliNB()
        clf.fit(features, target)  # Train only once
        # Evaluate model performance
        accuracy = clf.score(features, target)*100
        print("accuracy",accuracy)
        class_probabilities = clf.predict_proba(new_data)[0]
        print("class_probabilities",class_probabilities)
        predicted_class_index = class_probabilities.argmax()
        print("predicted_class_index",predicted_class_index)
        predicted_class = list(encoding_map['Play'].keys())[predicted_class_index]
        print("predicted_class",predicted_class)
        predicted_class_prob = class_probabilities[predicted_class_index] * 100
        print("predicted_class_prob",predicted_class_prob)
    
    # Calculate confusion matrix (if not already calculated)
    global confusion_matrix_result
    if not hasattr(predict_play, 'confusion_matrix_result'):
        confusion_matrix_result = confusion_matrix(target, clf.predict(features))
    print("confusion_matrix_result",confusion_matrix_result)
    sns.heatmap(confusion_matrix_result)  # Confusion matrix visualization
    # plt.show()  # Display the plots
    plt.savefig('5.jpg')

    # Predict the class for the new data
    prediction = clf.predict(new_data)
    print("prediction",prediction)
    predicted_class = list(encoding_map['Play'].keys())[list(encoding_map['Play'].values()).index(prediction[0])] if prediction[0] in encoding_map['Play'].values() else "Không biết"
    print("predicted_class",predicted_class)
    # result_path="2.jpg"
    # im = Image.open(result_path)
    return accuracy,precision,recall,f1,confusion_matrix_result,predicted_class #,im

    
interface = gr.Interface(
    predict_play, 
    inputs=[gr.Dropdown(label="Nhập thời tiết",choices=list(encoding_map['Outlook'].keys())),
            gr.Dropdown(label="Nhập vào Nhiệt độ",choices=list(encoding_map['Temp'].keys())),
            gr.Dropdown(label="Nhập vào độ ẩm",choices=list(encoding_map['Humidity'].keys())),
            gr.Dropdown(label="Nhập vào gió",choices=list(encoding_map['Windy'].keys()))],
    outputs=[gr.Text(label="Độ chính xác của mô hình(Model Accuracy):"),
             gr.Text(label="Kết quả Precision score:"),
             gr.Text(label="Kết quả Recall score:"),
             gr.Text(label="Kết quả F1 score:"),
             gr.Text(label="Kết quả confusion matrix:"),
             gr.Text(label="Kết quả Dự đoán:"),],
            #  gr.Image(label="Phân bố các loại thời tiết:"),],
    title="Dự báo người chơi",
    # examples=[["2.jpg"],
    #           ],
    description="Dự đoán xem hôm nay có chơi Golf hay không"
)

# Chạy giao diện Gradio
interface.launch(share=True)

