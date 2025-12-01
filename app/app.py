import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Simulador de Ventas Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #764ba2;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# FUNCIONES DE CARGA DE DATOS
# ============================================

@st.cache_resource
def cargar_modelo():
    """Carga el modelo de ML entrenado"""
    try:
        modelo = joblib.load('models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_datos():
    """Carga el dataframe de inferencia preparado"""
    try:
        # Cargar el archivo que tiene todas las columnas necesarias
        df = pd.read_csv('data/processed/df_nuevo.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Tomar datos de noviembre 2024 como base para noviembre 2025
        df_nov_2024 = df[(df['año'] == 2024) & (df['mes'] == 11)].copy()
        
        if df_nov_2024.empty:
            st.error("❌ No hay datos base para crear simulación")
            return None
        
        # Crear datos para noviembre 2025 basados en noviembre 2024
        df_nov = df_nov_2024.copy()
        
        # Actualizar el año y las fechas a 2025
        df_nov['año'] = 2025
        df_nov['fecha'] = df_nov['fecha'] + pd.DateOffset(years=1)
        
        return df_nov
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

# ============================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN RECURSIVA
# ============================================

def predecir_ventas_recursivas(df_producto, modelo, ajuste_descuento, escenario_competencia):
    """
    Realiza predicciones día por día con ajustes de precio y competencia
    
    Parámetros:
    - df_producto: DataFrame filtrado para un producto específico
    - modelo: Modelo de ML cargado
    - ajuste_descuento: Ajuste porcentual del descuento (-50 a +50)
    - escenario_competencia: Multiplicador del precio competencia (0.95, 1.0, 1.05)
    """
    
    # Hacer una copia para no modificar el original
    df = df_producto.copy()
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Obtener precio_base (es constante para el producto)
    precio_base = df['precio_base'].iloc[0]
    
    # Lista para almacenar predicciones
    predicciones = []
    
    # Recorrer cada día de noviembre
    for idx in range(len(df)):
        # ===== PASO 1: Recalcular variables dependientes =====
        
        # Calcular precio_venta con el descuento ajustado
        descuento_base = df.loc[idx, 'descuento_porcentaje']
        nuevo_descuento = descuento_base + ajuste_descuento
        nuevo_descuento = max(-50, min(50, nuevo_descuento))  # Limitar entre -50% y 50%
        
        df.loc[idx, 'descuento_porcentaje'] = nuevo_descuento
        df.loc[idx, 'precio_venta'] = precio_base * (1 + nuevo_descuento / 100)
        
        # Ajustar precio_competencia según escenario
        precio_comp_base = df.loc[idx, 'precio_competencia']
        df.loc[idx, 'precio_competencia'] = precio_comp_base * escenario_competencia
        
        # Recalcular ratio_precio
        if df.loc[idx, 'precio_competencia'] > 0:
            df.loc[idx, 'ratio_precio'] = df.loc[idx, 'precio_venta'] / df.loc[idx, 'precio_competencia']
        
        # ===== PASO 2: Hacer la predicción =====
        
        # Seleccionar las features que el modelo espera
        try:
            features = df.loc[[idx], modelo.feature_names_in_]
            prediccion = modelo.predict(features)[0]
            prediccion = max(0, prediccion)  # No puede ser negativa
            predicciones.append(prediccion)
        except Exception as e:
            st.error(f"Error en predicción del día {idx+1}: {e}")
            predicciones.append(0)
    
    # Añadir predicciones al dataframe
    df['unidades_predichas'] = predicciones
    df['ingresos_proyectados'] = df['unidades_predichas'] * df['precio_venta']
    
    return df

# ============================================
# FUNCIÓN DE COMPARATIVA DE ESCENARIOS
# ============================================

def comparar_escenarios(df_producto, modelo, ajuste_descuento):
    """Compara los 3 escenarios de competencia"""
    
    escenarios = {
        'Actual (0%)': 1.0,
        'Competencia -5%': 0.95,
        'Competencia +5%': 1.05
    }
    
    resultados = {}
    
    for nombre, multiplicador in escenarios.items():
        df_pred = predecir_ventas_recursivas(df_producto, modelo, ajuste_descuento, multiplicador)
        resultados[nombre] = {
            'unidades': df_pred['unidades_predichas'].sum(),
            'ingresos': df_pred['ingresos_proyectados'].sum()
        }
    
    return resultados

# ============================================
# CARGA INICIAL
# ============================================

modelo = cargar_modelo()
df_inferencia = cargar_datos()

if modelo is None or df_inferencia is None:
    st.stop()

# ============================================
# SIDEBAR - CONTROLES DE SIMULACIÓN
# ============================================

st.sidebar.title("🎯 Controles de Simulación")
st.sidebar.markdown("---")

# Obtener lista de productos únicos
productos = df_inferencia['nombre'].unique()
productos = sorted(productos)

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "📦 Seleccionar Producto",
    productos,
    index=0
)

# Slider de descuento
ajuste_descuento = st.sidebar.slider(
    "💰 Ajuste de Descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el descuento sobre el precio base. Valores negativos reducen el descuento, positivos lo aumentan."
)

# Selector de escenario de competencia
escenario_competencia_label = st.sidebar.radio(
    "🏪 Escenario de Competencia",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    help="Selecciona cómo varían los precios de la competencia"
)

# Mapeo de escenario a multiplicador
escenario_map = {
    "Actual (0%)": 1.0,
    "Competencia -5%": 0.95,
    "Competencia +5%": 1.05
}
escenario_multiplicador = escenario_map[escenario_competencia_label]

st.sidebar.markdown("---")

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Ajusta los controles y presiona 'Simular Ventas' para ver las proyecciones actualizadas.")

# ============================================
# ZONA PRINCIPAL - DASHBOARD
# ============================================

if simular:
    
    # Filtrar datos del producto seleccionado
    df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
    
    if df_producto.empty:
        st.error(f"❌ No hay datos para el producto: {producto_seleccionado}")
        st.stop()
    
    # ===== REALIZAR PREDICCIÓN CON SPINNER =====
    with st.spinner('🔮 Generando predicciones recursivas...'):
        df_resultados = predecir_ventas_recursivas(
            df_producto, 
            modelo, 
            ajuste_descuento, 
            escenario_multiplicador
        )
    
    # ===== HEADER =====
    st.title(f"📊 Dashboard de Ventas - {producto_seleccionado}")
    st.subheader("📅 Proyección Noviembre 2025")
    st.markdown("---")
    
    # ===== KPIs DESTACADOS =====
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unidades_totales = df_resultados['unidades_predichas'].sum()
        st.metric(
            label="🎯 Unidades Totales",
            value=f"{int(unidades_totales):,}",
            delta=None
        )
    
    with col2:
        ingresos_totales = df_resultados['ingresos_proyectados'].sum()
        st.metric(
            label="💰 Ingresos Proyectados",
            value=f"€{ingresos_totales:,.2f}",
            delta=None
        )
    
    with col3:
        precio_promedio = df_resultados['precio_venta'].mean()
        st.metric(
            label="💵 Precio Promedio",
            value=f"€{precio_promedio:.2f}",
            delta=None
        )
    
    with col4:
        descuento_promedio = df_resultados['descuento_porcentaje'].mean()
        st.metric(
            label="🏷️ Descuento Promedio",
            value=f"{descuento_promedio:.1f}%",
            delta=None
        )
    
    st.markdown("---")
    
    # ===== GRÁFICO DE PREDICCIÓN DIARIA =====
    st.subheader("📈 Predicción Diaria de Ventas - Noviembre 2025")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Configurar estilo de seaborn
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Gráfico de línea principal
    dias = df_resultados['dia_mes'].values
    unidades = df_resultados['unidades_predichas'].values
    
    ax.plot(dias, unidades, marker='o', linewidth=2.5, markersize=6, 
            color='#667eea', label='Unidades Predichas')
    
    # Marcar Black Friday (día 28)
    black_friday_idx = df_resultados[df_resultados['dia_mes'] == 28].index
    if len(black_friday_idx) > 0:
        bf_idx = black_friday_idx[0]
        bf_dia = df_resultados.loc[bf_idx, 'dia_mes']
        bf_unidades = df_resultados.loc[bf_idx, 'unidades_predichas']
        
        # Línea vertical
        ax.axvline(x=bf_dia, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Punto destacado
        ax.scatter([bf_dia], [bf_unidades], color='red', s=200, zorder=5, 
                   edgecolors='black', linewidth=2)
        
        # Anotación
        ax.annotate('🛍️ BLACK FRIDAY', 
                    xy=(bf_dia, bf_unidades), 
                    xytext=(bf_dia - 3, bf_unidades * 1.15),
                    fontsize=12, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Configuración del gráfico
    ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
    ax.set_title('Evolución de Ventas Diarias', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 31))
    
    # Ajustar límites del eje Y
    y_min = max(0, unidades.min() * 0.9)
    y_max = unidades.max() * 1.2
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # ===== TABLA DETALLADA =====
    st.subheader("📋 Detalle Diario de Ventas")
    
    # Preparar tabla para visualización
    df_tabla = df_resultados[['fecha', 'dia_mes', 'precio_venta', 'precio_competencia', 
                               'descuento_porcentaje', 'unidades_predichas', 
                               'ingresos_proyectados']].copy()
    
    # Añadir día de la semana
    df_tabla['dia_semana'] = df_tabla['fecha'].dt.day_name()
    
    # Reordenar columnas
    df_tabla = df_tabla[['fecha', 'dia_semana', 'precio_venta', 'precio_competencia',
                          'descuento_porcentaje', 'unidades_predichas', 'ingresos_proyectados']]
    
    # Renombrar columnas para mejor presentación
    df_tabla.columns = ['Fecha', 'Día Semana', 'Precio Venta (€)', 'Precio Competencia (€)',
                         'Descuento (%)', 'Unidades Predichas', 'Ingresos (€)']
    
    # Formatear números
    df_tabla['Precio Venta (€)'] = df_tabla['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
    df_tabla['Precio Competencia (€)'] = df_tabla['Precio Competencia (€)'].apply(lambda x: f"€{x:.2f}")
    df_tabla['Descuento (%)'] = df_tabla['Descuento (%)'].apply(lambda x: f"{x:.1f}%")
    df_tabla['Unidades Predichas'] = df_tabla['Unidades Predichas'].apply(lambda x: f"{int(x):,}")
    df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].apply(lambda x: f"€{x:.2f}")
    
    # Añadir emoji para Black Friday
    df_tabla['Fecha'] = df_tabla.apply(
        lambda row: f"🛍️ {row['Fecha']}" if '28' in str(row['Fecha']) else str(row['Fecha']),
        axis=1
    )
    
    # Mostrar tabla
    st.dataframe(df_tabla, width="stretch", hide_index=True)
    
    st.markdown("---")
    
    # ===== COMPARATIVA DE ESCENARIOS =====
    st.subheader("🔄 Comparativa de Escenarios de Competencia")
    st.markdown(f"*Manteniendo el ajuste de descuento en {ajuste_descuento:+d}%*")
    
    with st.spinner('📊 Comparando escenarios...'):
        resultados_escenarios = comparar_escenarios(df_producto, modelo, ajuste_descuento)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Actual (0%)")
        res = resultados_escenarios['Actual (0%)']
        st.metric("Unidades", f"{int(res['unidades']):,}")
        st.metric("Ingresos", f"€{res['ingresos']:,.2f}")
    
    with col2:
        st.markdown("### 📉 Competencia -5%")
        res = resultados_escenarios['Competencia -5%']
        st.metric("Unidades", f"{int(res['unidades']):,}")
        st.metric("Ingresos", f"€{res['ingresos']:,.2f}")
        st.caption("*Competencia baja precios 5%*")
    
    with col3:
        st.markdown("### 📈 Competencia +5%")
        res = resultados_escenarios['Competencia +5%']
        st.metric("Unidades", f"{int(res['unidades']):,}")
        st.metric("Ingresos", f"€{res['ingresos']:,.2f}")
        st.caption("*Competencia sube precios 5%*")
    
    st.markdown("---")
    
    # Mensaje de éxito
    st.success("✅ Simulación completada exitosamente")

else:
    # Pantalla inicial
    st.title("📊 Simulador de Ventas - Noviembre 2025")
    st.markdown("---")
    
    st.markdown("""
    ## 👋 Bienvenido al Simulador de Ventas
    
    Esta aplicación te permite simular y visualizar predicciones de ventas para noviembre 2025 
    utilizando un modelo de Machine Learning entrenado.
    
    ### 🚀 Cómo usar:
    
    1. **Selecciona un producto** del menú lateral
    2. **Ajusta el descuento** usando el slider (-50% a +50%)
    3. **Elige un escenario de competencia** (reducción, sin cambios, o aumento de precios)
    4. **Haz clic en "Simular Ventas"** para ver las proyecciones
    
    ### 📈 Verás:
    
    - **KPIs principales**: Unidades totales, ingresos, precios y descuentos promedio
    - **Gráfico de evolución**: Ventas día a día con Black Friday destacado
    - **Tabla detallada**: Información completa de cada día del mes
    - **Comparativa de escenarios**: Impacto de diferentes estrategias de competencia
    
    ### 🎯 Características:
    
    - ✅ Predicciones recursivas día por día
    - ✅ Actualización dinámica de lags y medias móviles
    - ✅ Visualización del Black Friday (28 de noviembre)
    - ✅ Comparativa de múltiples escenarios
    
    ---
    
    **💡 ¡Comienza ajustando los controles en el panel lateral!**
    """)
    
    # Mostrar información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write(f"**Tipo de modelo**: HistGradientBoostingRegressor")
        st.write(f"**Productos disponibles**: {len(productos)}")
        st.write(f"**Período de predicción**: Noviembre 2025 (30 días)")
        st.write(f"**Features utilizadas**: {len(modelo.feature_names_in_)}")
