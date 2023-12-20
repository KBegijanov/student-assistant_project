import random
import openpyxl
import csv
import nltk
from nltk.corpus import words
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import streamlit as st
from io import BytesIO
from xlsxwriter import Workbook
import joblib

# Загрузка словаря английских слов из библиотеки nltk
nltk.download('words')
english_words = set(words.words())
# Генерация списка из 10000 уникальных слов на английском языке
word_list = random.sample(list(english_words), 10000)
# Создаем список для столбца "Attempts" (число попыток от 6 до 10)
#attempts_list = [random.randint(5, 10) for num in range(10000)]
# Создаем список для столбца "Word Length" (количество букв в слове)
word_length_list = [len(word) for word in word_list]
# Создаем списки для столбцов "Consonants" и "Vowels" (количество согласных и гласных букв в слове)
consonants_list = [sum(1 for letter in word if letter.lower() in 'bcdfghjklmnpqrstvwxyz') for word in word_list]
vowels_list = [sum(1 for letter in word if letter.lower() in 'aeiou') for word in word_list]
# Записываем данные в CSV-файл
with open('sample.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #fieldnames = ['Word', 'Attempts', 'Word Length', 'Consonants', 'Vowels']
    fieldnames = ['Word', 'Word Length', 'Consonants', 'Vowels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(1000):
        writer.writerow({'Word': word_list[i],
                         #'Attempts': attempts_list[i],
                         'Word Length': word_length_list[i],
                         'Consonants': consonants_list[i],
                         'Vowels': vowels_list[i]})

# Загрузим данные из CSV-файла
data = pd.read_csv('sample.csv')

def calculate_attempts(row):
    if row['Word Length'] <= 3:
        attempts = 6
    elif 4 <= row['Word Length'] <= 6:
        attempts = 7
    elif 7 <= row['Word Length'] <= 8:
        attempts = 8
    elif row['Word Length'] >= 9:
        attempts = 9
    if row['Consonants'] >= 2 * row['Vowels']:
        attempts -= 1
    return attempts

data['Attempts'] = data.apply(calculate_attempts, axis=1)

# Создаем столбец "Real Attempts" и копируем значения из "Attempts"
data['Real Attempts'] = data['Attempts']

# Сохраняем таблицу в файл CSV
data.to_csv('sample.csv', index=False)

# Обновление значений в столбце "Real Attempts"
random_indices_increase = np.random.choice(data.index, size=int(0.35 * len(data)), replace=False)
random_indices_decrease = np.random.choice(data.index, size=int(0.25 * len(data)), replace=False)

data.loc[random_indices_increase, 'Real Attempts'] += 1.8
data.loc[random_indices_decrease, 'Real Attempts'] -= 0.8

# Подготовка данных для обучения
X = data[['Word Length', 'Vowels', 'Consonants']]
y = data['Real Attempts']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
#mse = mean_squared_error(y_test, y_pred)
#print(f'Mean Squared Error: {mse}')

joblib.dump(model, "model.pkl")
# Загрузка модели машинного обучения
model = joblib.load('model.pkl')

# Загрузка страницы
st.title("Прогнозирование количества повторений слов на английском языке для их запоминания")

# Загрузка таблицы пользователем
uploaded_file = st.file_uploader("Загрузите таблицу со словами на английском языке (Excel)", type=["xlsx", "xls"])
if uploaded_file is not None:
    user_df = pd.read_excel(uploaded_file, header=None, names=['Words'])

    # Прогнозирование чисел с использованием модели
    user_df['Word Length'] = user_df['Words'].apply(lambda x: len(x))
    user_df['Vowels'] = user_df['Words'].apply(lambda x: sum(1 for char in x if char.lower() in 'aeiou'))
    user_df['Consonants'] = user_df['Words'].apply(lambda x: sum(1 for char in x if char.lower() in 'bcdfghjklmnpqrstvwxyz'))

    X_user = user_df[['Word Length', 'Vowels', 'Consonants']]
    user_df['Predicted Attempts'] = model.predict(X_user)
    user_df['Predicted Attempts'] = user_df['Predicted Attempts'].round()

    # Группировка слов в зависимости от прогнозируемого числа
    grouped_df = user_df.groupby('Predicted Attempts')['Words'].apply(list).reset_index(name='Word List')

    # Вывод результата
    #st.write("Прогнозирование чисел для слов:")
    #st.write(user_df)

    st.write("Таблица слов на английском языке сгруппированная по количеству возрастания повторений для запоминания:")
    st.write(grouped_df)

    # Скачивание получившейся таблицы в формате Excel
   # st.write("Скачать результаты:")
    #st.write("1. Для полной таблицы")
    #st.download_button(
     #   label="Скачать полную таблицу",
      #  data=user_df.to_csv(index=False).encode('utf-8'),
       # file_name="predicted_table_full.csv",
        #mime="text/csv"
    #)

    st.write("Данные о необходимом количестве повторений слов для запоминания")
    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        grouped_df.to_excel(writer, index=False, sheet_name='Sheet1')
    st.download_button(
        label="Скачать данные",
        data=excel_data.getvalue(),
        file_name="predicted_table_grouped.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
