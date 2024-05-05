import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import pyaudio
import time
from difflib import SequenceMatcher

# Agregamos una nueva librería para análisis fonético
import epitran

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
