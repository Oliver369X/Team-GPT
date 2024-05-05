import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import pyaudio
import time
from difflib import SequenceMatcher
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Agregamos una nueva librería para análisis fonético
import epitran
pagina_principal = st.sidebar.radio("Navegar", ["Estudiante", "Docente"])

# Contenido para la pestaña "Inicio"
if pagina_principal == "Estudiante":
# Función para grabar y procesar el audio
    def record_audio():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Grabando, por favor lee el texto en voz alta...")
            start_time = time.time()
            audio = r.listen(source)
            duration = time.time() - start_time
            st.write("Grabación detenida, procesando...")
        try:
            text = r.recognize_google(audio, language="es-ES")  # Especificamos español
            st.write("Dijiste: " + text)
            return text, duration
        except sr.UnknownValueError:
            st.error("No pude entender el audio, intenta nuevamente.")
            return None, None
        except sr.RequestError as e:
            st.error(f"Error en el servicio de reconocimiento: {e}")
            return None, None

    # Función mejorada de evaluación de pronunciación
    def evaluate_pronunciation(text, reference_text):
        # Utilizamos Epitran para un análisis fonético más detallado
        epi = epitran.Epitran('spa-Latn')
        phonetic_text = epi.transliterate(text)
        phonetic_reference = epi.transliterate(reference_text)
        
        similarity = SequenceMatcher(None, phonetic_text, phonetic_reference).ratio()
        return f"Puntuación de pronunciación (similitud fonética): {similarity:.2f}/1.00"

    # Función mejorada de evaluación de fluidez
    def evaluate_fluency(text, duration):
        words = len(text.split())
        words_per_minute = words / duration * 60  # Palabras por minuto
        return f"Fluidez (palabras por minuto): {words_per_minute:.2f}"

    def main():
        st.sidebar.title("Configuración")
        openai_api_key = st.sidebar.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        reference_text = st.sidebar.text_area("Texto de referencia para la pronunciación", "Por favor, ingrese el texto que el usuario debería leer.")

        if not openai_api_key:
            st.sidebar.error("Por favor ingresa tu clave API de OpenAI.")
            st.stop()

        st.title("💬 Chatbot Educativo")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "¿Cómo puedo ayudarte?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("Grabar Audio"):
            text, duration = record_audio()
            if text:
                pronunciation_score = evaluate_pronunciation(text, reference_text)
                fluency_score = evaluate_fluency(text, duration)
                st.write(pronunciation_score)
                st.write(fluency_score)

                client = OpenAI(api_key=openai_api_key)
                st.session_state.messages.append({"role": "user", "content": text})
                st.chat_message("user").write(text)
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)

        if prompt := st.chat_input("Escribe aquí para continuar la conversación:"):
            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

    if __name__ == "__main__":
        main()
elif pagina_principal == "Docente":

    # Configuración para desactivar la advertencia sobre pyplot global
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Generar datos de ejemplo para los estudiantes
    nombres_estudiantes = ['Juan', 'María', 'Luis', 'Ana', 'Pedro', 'Laura', 'Carlos', 'Sofía', 'Diego', 'Elena', 'Miguel', 'Valentina', 'Andrés', 'Camila', 'David', 'Lucía', 'Fernando', 'Isabella', 'Javier', 'Paula']
    apellidos_estudiantes = ['Pérez', 'González', 'Gómez', 'Fernández', 'Martínez', 'Sánchez', 'Díaz', 'López', 'Rodríguez', 'Perez', 'Gonzalez', 'Gomez', 'Fernandez', 'Martinez', 'Sanchez', 'Diaz', 'Lopez', 'Rodriguez', 'Perez', 'Gonzalez']

    datos_estudiantes = pd.DataFrame({
        'Nombre': nombres_estudiantes,
        'Apellido': apellidos_estudiantes,
        'Progreso Lectura': np.random.randint(0, 101, 20),  # 0: se evalua de 0 al 100%
        'Tiempo de Lectura': np.random.randint(0, 61, 20),  # de 0 al 60 min
        'Visitas a la Página': np.random.randint(0, 101, 20)
    })

    # Título de la aplicación
    st.title('Monitoreo de Alumnos para Docentes')

    # Mostrar una tabla con el progreso de cada estudiante
    st.subheader('Progreso de los Estudiantes')

    # Mostrar la tabla con los datos de los estudiantes y su progreso
    st.dataframe(datos_estudiantes)

    # Agregar filtros para que el docente pueda seleccionar estudiantes específicos o ver el progreso en diferentes períodos de tiempo

    # Filtrar por nombre de estudiante
    nombre_estudiante = st.sidebar.selectbox('Seleccionar Estudiante', datos_estudiantes['Nombre'])

    # Filtrar por período de tiempo
    periodo_tiempo = st.sidebar.selectbox('Seleccionar Período de Tiempo', ['Hoy','Última Semana', 'Último Mes', 'Último Trimestre'])

    # Generar datos de ejemplo para el progreso de los estudiantes
    datos_progreso = []
    dias_mes = pd.date_range(start='2024-05-01', end='2024-05-31')
    for estudiante in nombres_estudiantes:
        for dia in dias_mes:
            # Simular actividades aleatorias para cada día
            progreso_lectura = np.random.randint(0, 101)  # 0: se evalua de 0 al 100%
            tiempo_lectura = np.random.randint(0, 61)  # de 0 al 60 min
            visitas_pagina = np.random.randint(0, 101) 
            
            # Crear un diccionario con los datos de cada día para el estudiante
            datos_dia = {
                'Nombre': estudiante,
                'Día': dia,
                'Progreso Lectura': progreso_lectura,
                'Tiempo de Lectura': tiempo_lectura,
                'Visitas a la Página': visitas_pagina
            }
            
            datos_progreso.append(datos_dia)
    df_progreso = pd.DataFrame(datos_progreso)

    # Convertir la columna 'Día' a tipo Timestamp
    df_progreso['Día'] = pd.to_datetime(df_progreso['Día'])

    # Filtrar datos según el período de tiempo seleccionado
    if periodo_tiempo == 'Hoy':
        fecha_hoy = pd.Timestamp.today().floor('D')
        df_progreso_filtrado = df_progreso[df_progreso['Día'] == fecha_hoy]
    elif periodo_tiempo == 'Última Semana':
        fecha_inicio_semana = pd.Timestamp.today().floor('D') - pd.Timedelta(days=pd.Timestamp.today().weekday()) - pd.Timedelta(weeks=1)
        df_progreso_filtrado = df_progreso[(df_progreso['Día'] >= fecha_inicio_semana) & (df_progreso['Día'] <= pd.Timestamp.today().floor('D'))]
    elif periodo_tiempo == 'Último Mes':
        fecha_inicio_mes = pd.Timestamp.today().floor('D') - pd.Timedelta(days=pd.Timestamp.today().day - 1) - pd.Timedelta(days=30)
        df_progreso_filtrado = df_progreso[(df_progreso['Día'] >= fecha_inicio_mes) & (df_progreso['Día'] <= pd.Timestamp.today().floor('D'))]
    else:
        fecha_inicio_trimestre = pd.Timestamp.today().floor('D') - pd.Timedelta(days=pd.Timestamp.today().day - 1) - pd.Timedelta(days=90)
        df_progreso_filtrado = df_progreso[(df_progreso['Día'] >= fecha_inicio_trimestre) & (df_progreso['Día'] <= pd.Timestamp.today().floor('D'))]

    # Mostrar gráficos o visualizaciones del progreso de los estudiantes

    # Gráfico de barras o líneas del progreso de lectura por estudiante
    st.subheader('Progreso de Lectura por Estudiante')
    tipo_grafico_lectura = st.selectbox('Seleccionar Tipo de Gráfico', ['Gráfico de Barras', 'Gráfico de Líneas'])
    grafico_data_lectura = df_progreso_filtrado.loc[df_progreso_filtrado['Nombre'] == nombre_estudiante, ['Día', 'Progreso Lectura']]

    if tipo_grafico_lectura == 'Gráfico de Barras':
        plt.figure(figsize=(10,6))
        plt.bar(grafico_data_lectura['Día'], grafico_data_lectura['Progreso Lectura'], color='blue')
        plt.title('Progreso de Lectura por Estudiante (Gráfico de Barras)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Progreso de Lectura', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot()
    else:
        plt.figure(figsize=(10,6))
        plt.plot(grafico_data_lectura['Día'], grafico_data_lectura['Progreso Lectura'], color='blue', marker='o', linestyle='-')
        plt.title('Progreso de Lectura por Estudiante (Gráfico de Líneas)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Progreso de Lectura', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot()

    # Selector de tipo de gráfico para Tiempo de Lectura
    st.subheader('Tiempo de Lectura por Estudiante')
    tipo_grafico_tiempo_key = 'tipo_grafico_tiempo'  # Clave única para el selector de tipo de gráfico de tiempo
    tipo_grafico_tiempo = st.selectbox('Seleccionar Tipo de Gráfico', ['Gráfico de Barras', 'Gráfico de Líneas'], key=tipo_grafico_tiempo_key)
    grafico_data_tiempo = df_progreso_filtrado.loc[df_progreso_filtrado['Nombre'] == nombre_estudiante, ['Día', 'Tiempo de Lectura']]

    if tipo_grafico_tiempo == 'Gráfico de Barras':
        plt.figure(figsize=(10,6))
        plt.bar(grafico_data_tiempo['Día'], grafico_data_tiempo['Tiempo de Lectura'], color='green')
        plt.title('Tiempo de Lectura por Estudiante (Gráfico de Barras)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Tiempo de Lectura', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot()
    else:
        plt.figure(figsize=(10,6))
        plt.plot(grafico_data_tiempo['Día'], grafico_data_tiempo['Tiempo de Lectura'], color='green', marker='o', linestyle='-')
        plt.title('Tiempo de Lectura por Estudiante (Gráfico de Líneas)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Tiempo de Lectura', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot()

    # Selector de tipo de gráfico para Visitas a la Página
    st.subheader('Visitas a la Página por Estudiante')
    tipo_grafico_pagina_key = 'tipo_grafico_pagina'  # Clave única para el selector de tipo de gráfico de páginas visitadas
    tipo_grafico_pagina = st.selectbox('Seleccionar Tipo de Gráfico', ['Gráfico de Barras', 'Gráfico de Líneas'], key=tipo_grafico_pagina_key)
    grafico_data_pagina = df_progreso_filtrado.loc[df_progreso_filtrado['Nombre'] == nombre_estudiante, ['Día', 'Visitas a la Página']]

    if tipo_grafico_pagina == 'Gráfico de Barras':
        plt.figure(figsize=(10,6))
        plt.bar(grafico_data_pagina['Día'], grafico_data_pagina['Visitas a la Página'], color='orange')
        plt.title('Visitas a la Página por Estudiante (Gráfico de Barras)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Visitas a la Página', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot()
    else:
        plt.figure(figsize=(10,6))
        plt.plot(grafico_data_pagina['Día'], grafico_data_pagina['Visitas a la Página'], color='orange', marker='o', linestyle='-')
        plt.title('Visitas a la Página por Estudiante (Gráfico de Líneas)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Visitas a la Página', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot()

    # Agregar resúmenes de estadísticas sobre el progreso general de los estudiantes

    # Resumen de estadísticas sobre el progreso de lectura
    st.sidebar.subheader('Resumen de Progreso de Lectura')
    promedio_lectura = df_progreso_filtrado['Progreso Lectura'].mean()
    maximo_lectura = df_progreso_filtrado['Progreso Lectura'].max()
    minimo_lectura = df_progreso_filtrado['Progreso Lectura'].min()
    st.sidebar.write(f'Promedio de Progreso de Lectura: {promedio_lectura}')
    st.sidebar.write(f'Máximo Progreso de Lectura: {maximo_lectura}')
    st.sidebar.write(f'Mínimo Progreso de Lectura: {minimo_lectura}')

    # Resumen de estadísticas sobre el progreso de tiempo de lectura
    st.sidebar.subheader('Resumen de Tiempo de Lectura')
    promedio_tiempo_lectura = df_progreso_filtrado['Tiempo de Lectura'].mean()
    maximo_tiempo_lectura = df_progreso_filtrado['Tiempo de Lectura'].max()
    minimo_tiempo_lectura = df_progreso_filtrado['Tiempo de Lectura'].min()
    st.sidebar.write(f'Promedio de Tiempo de Lectura: {promedio_tiempo_lectura}')
    st.sidebar.write(f'Máximo Tiempo de Lectura: {maximo_tiempo_lectura}')
    st.sidebar.write(f'Mínimo Tiempo de Lectura: {minimo_tiempo_lectura}')

    # Resumen de estadísticas sobre el uso de la página
    st.sidebar.subheader('Resumen de Uso de la Página')
    promedio_uso_pagina = df_progreso_filtrado['Visitas a la Página'].mean()
    maximo_uso_pagina = df_progreso_filtrado['Visitas a la Página'].max()
    minimo_uso_pagina = df_progreso_filtrado['Visitas a la Página'].min()
    st.sidebar.write(f'Promedio de Visitas a la Página: {promedio_uso_pagina}')
    st.sidebar.write(f'Máximo Visitas a la Página: {maximo_uso_pagina}')
    st.sidebar.write(f'Mínimo Visitas a la Página: {minimo_uso_pagina}')
