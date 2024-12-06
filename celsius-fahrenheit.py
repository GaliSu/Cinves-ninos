import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import time  

#Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Interfaz de usuario con Streamlit
st.title("Entrenamiento de RN para Conversión: Celsius-Fahrenheit") #titulo

#Deslizadores interactivos
epocas = st.slider("Épocas", min_value=10, max_value=500, step=10, value=100) #slide epocas
tasa_aprendizaje = st.slider("Tasa de aprendizaje", min_value=0.001, max_value=0.5, step=0.01, value=0.1) #slide tasa de aprendizaje
perdida = st.selectbox("Función de pérdida", ["mean_squared_error", "mean_absolute_error"]) #caja selección de tipo de perdida

#Controles para el número de capas y neuronas por capa
num_capas = st.slider("Número de capas", min_value=1, max_value=10, step=1, value=3) #slide numero de capas
neurona_por_capa = st.slider("Número de neuronas por capa", min_value=1, max_value=20, step=1, value=4) #slide numero de neuronas

#Función para visualizar la rn con Plotly
def visualizar_red(num_capas, neurona_por_capa):
    fig = go.Figure() #Crea un objeto Figure vacío donde se agregarán las capas y conexiones

    # Añadir capas con las neuronas
    for capa in range(num_capas):
        for neurona in range(neurona_por_capa):
            fig.add_trace(go.Scatter(
                x=[capa], y=[neurona], mode="markers", marker=dict(size=15),
                name=f"Capa {capa + 1}, Neurona {neurona + 1}",
                text=f"Capa {capa + 1}, Neurona {neurona + 1}",
                showlegend=False
            ))

        #Conectamos las neuronas de cada capa con líneas
        if capa > 0:
            for neurona_anterior in range(neurona_por_capa):
                for neurona_actual in range(neurona_por_capa):
                    fig.add_trace(go.Scatter(
                        x=[capa-1, capa], y=[neurona_anterior, neurona_actual],
                        mode="lines", line=dict(color="gray", width=1),
                        showlegend=False
                    ))

    fig.update_layout( #Configuración del diseño del grafico
        title="Red Neuronal Interactiva",
        xaxis=dict(title="Capas", range=[-1, num_capas]),
        yaxis=dict(title="Neurona", range=[-1, neurona_por_capa]),
        showlegend=False,
        plot_bgcolor="white",
        width=800,
        height=500
    )
    
    st.plotly_chart(fig)#Muestra el grafico de la rn interactivo

#Mostrar rn con la configuración actual
visualizar_red(num_capas, neurona_por_capa)

#Botón para entrenar el modelo
if st.button("Entrenar modelo"):
    #Crear las capas dinámicamente según la entrada del usuario
    capas = [tf.keras.layers.Dense(units=neurona_por_capa, input_shape=[1], activation='relu') for _ in range(num_capas)]
    
    #Añadir la capa de salida
    capas.append(tf.keras.layers.Dense(1)) #una sola neurona(predecir una unica variable)
    
    #Crear el modelo
    modelo = tf.keras.Sequential(capas) #modelo secuencial
    modelo.compile(optimizer=tf.keras.optimizers.Adam(tasa_aprendizaje), loss=perdida) #compila el modelo

    #Contenedor vacío para la actualización en tiempo real
    grafica_dispersion = st.empty()
    
    #Ciclo de entrenamiento
    for epoca in range(epocas):
        #Entrena el modelo usando los datos de enatrada
        historial = modelo.fit(celsius, fahrenheit, epochs=1, verbose=0)
        
        #Predicción para cada época
        predicciones = modelo.predict(celsius)
        
        # Crear la gráfica de dispersión con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=celsius, y=fahrenheit, mode='markers', name='Datos Reales', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=celsius, y=predicciones.flatten(), mode='lines', name='Predicción del Modelo', line=dict(color='red')))
        
        #Personaliza la gráfica
        fig.update_layout(
            title=f"Época {epoca + 1} - Predicción de Celsius a Fahrenheit",
            xaxis_title="Celsius",
            yaxis_title="Fahrenheit",
            plot_bgcolor="white"
        )
        
        # Mostrar la gráfica en tiempo real
        grafica_dispersion.plotly_chart(fig)
        
        #Timer(pausa)
        time.sleep(0.05)  

    # Mostrar la predicción final
    st.write(f"Predicción final después de {epocas} épocas:")
    for i, cel in enumerate(celsius):
        st.write(f"{cel}°C = {predicciones[i][0]:.2f}°F (real: {fahrenheit[i]}°F)")
