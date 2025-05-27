import pandas as pd
from gurobipy import GRB, Model
from typing import Dict, List, Any


def cargar_datos() -> Dict[str, Any]:
    """
    Esta función carga los archivos .csv de la instancia y devuelve
    un diccionario con todos los parámetros necesarios para construir el modelo.
    
    Returns:
        Dict[str, Any]: Diccionario con todos los parámetros del problema
    """
    # Cargar conjuntos básicos
    areas_verdes = pd.read_csv    # 19. Si se decide instalar un riego k, entonces es el riego escogido
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        R[i, j, k] >= Z_plus[i, j, k],
                        f"instalar_implica_escogido_{i}_{j}_{k}"
                    )
                    
    # Restricción adicional: Un sistema se utiliza solo si se instala o se mantiene
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        R[i, j, k] <= Z[i, j, k] + Z_plus[i, j, k],
                        f"escogido_requiere_mantener_o_instalar_{i}_{j}_{k}"
                    )eas_verdes.csv")
    plantas = pd.read_csv("data/plantas.csv")
    sistemas_riego = pd.read_csv("data/sistemas_riego.csv")
    dias = pd.read_csv("data/dias.csv")
    
    # Cargar parámetros
    area_planta = pd.read_csv("data/area_planta.csv")
    requerimiento_agua = pd.read_csv("data/requerimiento_agua_planta.csv")
    frecuencia_max = pd.read_csv("data/frecuencia_maxima_planta.csv")
    frecuencia_min = pd.read_csv("data/frecuencia_minima_planta.csv")
    eficiencia = pd.read_csv("data/eficiencia_sistema_planta.csv")
    agua_natural = pd.read_csv("data/agua_natural.csv")
    agua_disponible = pd.read_csv("data/agua_disponible.csv")
    sistemas_instalados = pd.read_csv("data/sistemas_instalados.csv")
    costo_instalacion = pd.read_csv("data/costo_instalacion.csv")
    costo_desinstalacion = pd.read_csv("data/costo_desinstalacion.csv")
    costo_mantenimiento = pd.read_csv("data/costo_mantenimiento.csv")
    frecuencia_max_mantenimiento = pd.read_csv("data/frecuencia_max_mantenimiento.csv")
    presupuesto = pd.read_csv("data/presupuesto.csv")
    
    # Crear conjuntos (listas de índices)
    areas_set = areas_verdes['id_area'].tolist()  # Conjunto de áreas verdes
    plantas_set = plantas['id_planta'].tolist()  # Conjunto de plantas
    sistemas_set = sistemas_riego['id_sistema'].tolist()  # Conjunto de sistemas de riego
    dias_set = dias['id_dia'].tolist()  # Conjunto de días
    
    # Crear parámetros
    a_ij = {(row['id_area'], row['id_planta']): row['area_m2'] 
            for _, row in area_planta.iterrows()}
    
    b_j = {row['id_planta']: row['litros_m2'] 
           for _, row in requerimiento_agua.iterrows()}
    
    f_max_j = {row['id_planta']: row['frecuencia_max_dias'] 
               for _, row in frecuencia_max.iterrows()}
    
    f_min_j = {row['id_planta']: row['frecuencia_min_dias'] 
               for _, row in frecuencia_min.iterrows()}
    
    # Create efficiency parameter e_jk with index order matching the mathematical model (j,k)
    # Data is read from CSV with id_sistema first, then id_planta, but we'll index it as [j,k] for the mathematical notation
    e_jk = {(row['id_planta'], row['id_sistema']): row['eficiencia'] 
            for _, row in eficiencia.iterrows()}
    
    w_tij = {(row['id_dia'], row['id_area'], row['id_planta']): row['vol_agua_natural_litros'] 
             for _, row in agua_natural.iterrows()}
    
    # Calcular parámetro v_ijt (1 si la planta se considera regada por agua natural, 0 EOC)
    v_ijt = {}
    for _, row in agua_natural.iterrows():
        t, i, j = row['id_dia'], row['id_area'], row['id_planta']
        if j in b_j:
            v_ijt[(i, j, t)] = 1 if row['vol_agua_natural_litros'] >= b_j[j] else 0
        else:
            v_ijt[(i, j, t)] = 0
    
    q_t = {row['id_dia']: row['agua_disponible_litros'] 
           for _, row in agua_disponible.iterrows()}
    
    s_ijk = {(row['id_area'], row['id_planta'], row['id_sistema']): row['existe_previamente'] 
             for _, row in sistemas_instalados.iterrows()}
    
    g_k = {row['id_sistema']: row['costo_instalacion_m2'] 
           for _, row in costo_instalacion.iterrows()}
    
    d_k = {row['id_sistema']: row['costo_desinstalacion_m2'] 
           for _, row in costo_desinstalacion.iterrows()}
    
    c_k = {row['id_sistema']: row['costo_mantenimiento_m2'] 
           for _, row in costo_mantenimiento.iterrows()}
    
    l_k = {row['id_sistema']: row['frecuencia_max_mantenimiento'] 
           for _, row in frecuencia_max_mantenimiento.iterrows()}
    
    p = presupuesto['presupuesto'].iloc[0]
    
    # Crear diccionario de datos
    data = {
        'areas_set': areas_set,  # Conjunto de áreas verdes
        'plantas_set': plantas_set,  # Conjunto de plantas
        'sistemas_set': sistemas_set,  # Conjunto de sistemas de riego
        'dias_set': dias_set,  # Conjunto de días
        'a_ij': a_ij,  # Área en m^2 de planta j en área verde i
        'b_j': b_j,  # Volumen de agua en litros que requiere la planta j por m^2
        'f_max_j': f_max_j,  # Frecuencia máxima de riego en días para planta j
        'f_min_j': f_min_j,  # Frecuencia mínima de riego en días para planta j
        'e_jk': e_jk,  # Eficiencia del sistema k para la planta j (indexado como [j,k] para seguir notación matemática)
        'w_tij': w_tij,  # Volumen de agua natural en día t para planta j en área verde i
        'v_ijt': v_ijt,  # Indicador si la planta j en área i en día t se considera regada por agua natural
        'q_t': q_t,  # Volumen de agua disponible para riego en día t
        's_ijk': s_ijk,  # Indicador si existe previamente sistema k para planta j en área i
        'g_k': g_k,  # Costo instalación de sistema k por m^2
        'd_k': d_k,  # Costo desinstalación de sistema k por m^2
        'c_k': c_k,  # Costo mantenimiento de sistema k por m^2
        'l_k': l_k,  # Frecuencia máxima de mantenimiento de sistema k en días
        'p': p,  # Presupuesto municipal
    }
    
    return data


def construir_modelo(data: Dict[str, Any]) -> Model:
    """
    Esta función construye el modelo de optimización utilizando Gurobi
    y los datos provistos en el diccionario `data`.
    
    Args:
        data: Diccionario con los conjuntos y parámetros necesarios para construir el modelo.
        
    Returns:
        Model: Modelo de optimización Gurobi configurado.
    """
    # Extraer datos con nombres descriptivos
    areas_set = data['areas_set']  # Conjunto de áreas verdes
    plantas_set = data['plantas_set']  # Conjunto de plantas
    sistemas_set = data['sistemas_set']  # Conjunto de sistemas de riego
    dias_set = data['dias_set']  # Conjunto de días
    
    # Extraer parámetros
    a_ij = data['a_ij']  # Área en m^2 de planta j en área verde i
    b_j = data['b_j']    # Volumen de agua en litros que requiere la planta j por m^2
    f_max_j = data['f_max_j']  # Frecuencia máxima de riego en días para planta j
    f_min_j = data['f_min_j']  # Frecuencia mínima de riego en días para planta j
    e_jk = data['e_jk']   # Eficiencia del sistema k para la planta j (indexado como [j,k])
    w_tij = data['w_tij'] # Volumen de agua natural en día t para planta j en área verde i
    v_ijt = data['v_ijt'] # Indicador si la planta j en área i en día t se considera regada por agua natural
    q_t = data['q_t']     # Volumen de agua disponible para riego en día t
    s_ijk = data['s_ijk'] # Indicador si existe previamente sistema k para planta j en área i
    g_k = data['g_k']     # Costo instalación de sistema k por m^2
    d_k = data['d_k']     # Costo desinstalación de sistema k por m^2
    c_k = data['c_k']     # Costo mantenimiento de sistema k por m^2
    l_k = data['l_k']     # Frecuencia máxima de mantenimiento de sistema k en días
    p = data['p']         # Presupuesto municipal
    
    # Crear el modelo
    model = Model("Optimizacion_Sistemas_Riego")  # type: ignore
    
    # Crear variables de decisión
    
    # X_ijtk: volumen de agua para planta j en área i en día t mediante sistema k
    X = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:  # Solo si la planta j existe en el área verde i
                for t in dias_set:
                    for k in sistemas_set:
                        X[i, j, t, k] = model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=0,
                            name=f"X_{i}_{j}_{t}_{k}"
                        )
    
    # Y_ijtk: indicador si se riega planta j en área i en día t mediante sistema k
    Y = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for t in dias_set:
                    for k in sistemas_set:
                        Y[i, j, t, k] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"Y_{i}_{j}_{t}_{k}"
                        )
    
    # R_ijk: indicador si se utiliza sistema k para planta j en área i
    R = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    R[i, j, k] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"R_{i}_{j}_{k}"
                    )
    
    # Z_ijk^-: indicador si se desinstala sistema k para planta j en área i
    Z_minus = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    Z_minus[i, j, k] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Z_minus_{i}_{j}_{k}"
                    )
    
    # Z_ijk: indicador si se mantiene sistema k para planta j en área i
    Z = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    Z[i, j, k] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Z_{i}_{j}_{k}"
                    )
    
    # Z_ijk^+: indicador si se instala sistema k para planta j en área i
    Z_plus = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    Z_plus[i, j, k] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Z_plus_{i}_{j}_{k}"
                    )
    
    # M_ijkt: indicador si se realiza mantenimiento del sistema k para planta j en área i en día t
    M = {}
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    for t in dias_set:
                        M[i, j, k, t] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"M_{i}_{j}_{k}_{t}"
                        )
    
    # Actualizar el modelo con variables
    model.update()
    
    # Valor grande M para restricciones
    big_M = 100000  # Un valor suficientemente grande
    
    # Restricciones
    
    # 1. Cumplir con el requerimiento hídrico de la planta j
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for t in dias_set:
                    for k in sistemas_set:
                        model.addConstr(
                            b_j[j] * a_ij[i, j] * Y[i, j, t, k] <= 
                            (w_tij.get((t, i, j), 0) + X[i, j, t, k] * e_jk[j, k]),
                            f"req_hidrico_{i}_{j}_{t}_{k}"
                        )
    
    # 2. Cumplir con la frecuencia mínima de la planta j
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                f_min = f_min_j[j]
                for t in dias_set:
                    if t >= f_min:  # Solo para días donde se puede verificar la frecuencia mínima
                        for k in sistemas_set:
                            # Suma de riegos (propios y naturales) en los días anteriores
                            suma_riegos = 0
                            for t_prev in range(max(1, t - f_min + 1), t):
                                if t_prev in dias_set:
                                    suma_riegos += Y[i, j, t_prev, k]
                                    if (i, j, t_prev) in v_ijt:
                                        suma_riegos += v_ijt[i, j, t_prev]
                            
                            model.addConstr(
                                big_M * (1 - Y[i, j, t, k]) >= suma_riegos,
                                f"freq_min_{i}_{j}_{t}_{k}"
                            )
    
    # 3. Cumplir con la frecuencia máxima de la planta j
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                f_max = f_max_j[j]
                for t in dias_set:
                    if t >= f_max:  # Solo para días donde se puede verificar la frecuencia máxima
                        for k in sistemas_set:
                            # Suma de riegos (propios y naturales) en los días anteriores
                            suma_riegos = 0
                            for t_prev in range(max(1, t - f_max + 1), t):
                                if t_prev in dias_set:
                                    suma_riegos += Y[i, j, t_prev, k]
                                    if (i, j, t_prev) in v_ijt:
                                        suma_riegos += v_ijt[i, j, t_prev]
                            
                            model.addConstr(
                                1 - Y[i, j, t, k] <= suma_riegos,
                                f"freq_max_{i}_{j}_{t}_{k}"
                            )
    
    # 4. Si una planta j se considera regada por lluvia, entonces no se riega artificialmente
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for t in dias_set:
                    if (i, j, t) in v_ijt and v_ijt[i, j, t] == 1:
                        for k in sistemas_set:
                            model.addConstr(
                                Y[i, j, t, k] == 0,
                                f"no_regar_lluvia_{i}_{j}_{t}_{k}"
                            )
    
    # 5. Si no se riega una planta j entonces el agua destinada a riego es 0
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for t in dias_set:
                    for k in sistemas_set:
                        model.addConstr(
                            X[i, j, t, k] <= big_M * Y[i, j, t, k],
                            f"agua_cero_{i}_{j}_{t}_{k}"
                        )
    
    # 6. Escoger el sistema de riego k con una eficiencia mínima de 0.8 con j
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        e_jk[j, k] >= 0.8 * R[i, j, k],
                        f"eficiencia_min_{i}_{j}_{k}"
                    )
    
    # 7. Si no se encuentra implementado el sistema de riego k, entonces no se riega mediante tal sistema
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for t in dias_set:
                    for k in sistemas_set:
                        model.addConstr(
                            Y[i, j, t, k] <= R[i, j, k],
                            f"sistema_implementado_{i}_{j}_{t}_{k}"
                        )
    
    # 8. No sobrepasar el volumen de agua disponible para riegos
    for t in dias_set:
        model.addConstr(
            sum(X[i, j, t, k] for i in areas_set for j in plantas_set for k in sistemas_set if (i, j) in a_ij) <= q_t[t],
            f"agua_disponible_{t}"
        )
    
    # 9. Se debe escoger uno y solo un sistema de riego por planta j en el área verde i
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                model.addConstr(
                    sum(R[i, j, k] for k in sistemas_set) == 1,
                    f"un_sistema_{i}_{j}"
                )
    
    # 10. Solo se puede mantener un sistema de riego si estaba previamente instalado
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    s_val = s_ijk.get((i, j, k), 0)
                    model.addConstr(
                        Z[i, j, k] <= s_val,
                        f"mantener_instalado_{i}_{j}_{k}"
                    )
    
    # 11. No se puede instalar y mantener el mismo sistema de riego
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        Z[i, j, k] + Z_plus[i, j, k] <= 1,
                        f"no_instalar_mantener_{i}_{j}_{k}"
                    )
    
    # 12. No se puede desinstalar y mantener el mismo sistema de riego
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        Z[i, j, k] + Z_minus[i, j, k] <= 1,
                        f"no_desinstalar_mantener_{i}_{j}_{k}"
                    )
    
    # 13. No se puede desinstalar e instalar el mismo sistema de riego
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        Z_minus[i, j, k] + Z_plus[i, j, k] <= 1,
                        f"no_desinstalar_instalar_{i}_{j}_{k}"
                    )
    
    # 14. Si es que se decide instalar un sistema de riego k, debe ser solo uno
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                model.addConstr(
                    sum(Z_plus[i, j, k] for k in sistemas_set) <= 1,
                    f"instalar_uno_{i}_{j}"
                )
    
    # 15. Si es que se decide desinstalar un sistema de riego k, debe ser solo uno
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                model.addConstr(
                    sum(Z_minus[i, j, k] for k in sistemas_set) <= 1,
                    f"desinstalar_uno_{i}_{j}"
                )
    
    # 16. Si se decide mantener un riego k, entonces no se desinstala nada
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        sum(Z_minus[i, j, k_prime] for k_prime in sistemas_set) == 1 - Z[i, j, k],
                        f"mantener_no_desinstalar_{i}_{j}_{k}"
                    )

    # 17. Si se decide mantener un riego k, entonces no se instala nada
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        sum(Z_plus[i, j, k_prime] for k_prime in sistemas_set) == 1 - Z[i, j, k],
                        f"mantener_no_instalar_{i}_{j}_{k}"
                    )
    
    # 18. Si se decide mantener un riego k, entonces es el riego escogido
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        R[i, j, k] >= Z[i, j, k],
                        f"mantener_implica_escogido_{i}_{j}_{k}"
                    )
    
    # 19. Si se decide instalar un riego k, entonces es el riego escogido
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        R[i, j, k] >= Z_plus[i, j, k],
                        f"instalar_implica_escogido_{i}_{j}_{k}"
                    )
    
    # Constraint adicional: Si se utiliza un sistema k, es porque se mantiene o se instala
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    model.addConstr(
                        R[i, j, k] <= Z[i, j, k] + Z_plus[i, j, k],
                        f"escogido_requiere_mantener_o_instalar_{i}_{j}_{k}"
                    )
    
    # 20. Se debe respetar la frecuencia máxima de mantenimiento del sistema de riego k
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    for t in dias_set:
                        if t > 1:  # No aplicar para el primer día
                            # Suma de usos en días anteriores
                            suma_usos = sum(Y[i, j, t_prev, k] for t_prev in range(1, t) if t_prev in dias_set)
                            # Suma de mantenimientos en días anteriores
                            suma_mantto = sum(M[i, j, k, t_prev] for t_prev in range(1, t) if t_prev in dias_set)
                            
                            model.addConstr(
                                suma_usos - l_k[k] * suma_mantto <= (l_k[k] - 1) + M[i, j, k, t],
                                f"freq_mantto_{i}_{j}_{k}_{t}"
                            )
    
    # 21. Solo se puede realizar mantenimiento a los sistemas de riego k implementados
    for i in areas_set:
        for j in plantas_set:
            if (i, j) in a_ij:
                for k in sistemas_set:
                    for t in dias_set:
                        model.addConstr(
                            M[i, j, k, t] <= R[i, j, k],
                            f"mantto_implementado_{i}_{j}_{k}_{t}"
                        )
    
    # 22. Se debe respetar el presupuesto municipal p
    model.addConstr(
        sum(Z_minus[i, j, k] * a_ij[i, j] * d_k[k] for i in areas_set for j in plantas_set for k in sistemas_set if (i, j) in a_ij) +
        sum(Z_plus[i, j, k] * a_ij[i, j] * g_k[k] for i in areas_set for j in plantas_set for k in sistemas_set if (i, j) in a_ij) +
        sum(M[i, j, k, t] * a_ij[i, j] * c_k[k] for i in areas_set for j in plantas_set for k in sistemas_set for t in dias_set if (i, j) in a_ij)
        <= p,
        "presupuesto"
    )
    
    # Función objetivo: minimizar el agua utilizada
    model.setObjective(
        sum(X[i, j, t, k] for i in areas_set for j in plantas_set for t in dias_set for k in sistemas_set if (i, j) in a_ij),
        GRB.MINIMIZE
    )
    
    return model


def resolver_modelo(model: Model) -> Model:
    """
    Esta función llama al solver de Gurobi para resolver el modelo.
    
    Args:
        model: Modelo de optimización Gurobi a resolver.
        
    Returns:
        Model: Modelo de optimización Gurobi ya resuelto.
    """
    # Configuración del solver
    model.setParam('TimeLimit', 600)  # Límite de tiempo en segundos (10 minutos)
    model.setParam('MIPGap', 0.01)    # Gap de optimalidad (1%)
    
    # Resolver el modelo
    model.optimize()
    
    return model


def imprimir_resultados(model: Model) -> None:
    """
    Esta función imprime de forma clara el valor óptimo y los resultados del modelo.
    
    Args:
        model: Modelo de optimización Gurobi ya resuelto.
    """
    # Verificar si el modelo se resolvió exitosamente
    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT and model.status != GRB.SUBOPTIMAL:
        print("El modelo no se pudo resolver de manera óptima.")
        print(f"Estado: {model.status}")
        return
    
    # Cargar datos para contextualizar resultados
    data = cargar_datos()
    # Obtenemos los datos necesarios del diccionario
    a_ij = data['a_ij']  # Área en m^2 de planta j en área verde i
    
    print("\n=== RESULTADOS DE LA OPTIMIZACIÓN ===")
    print(f"Volumen total de agua utilizada: {model.objVal:.2f} litros")
    
    # Extraer variables para análisis
    X_vars = {}
    Y_vars = {}
    R_vars = {}
    Z_plus_vars = {}
    Z_minus_vars = {}
    Z_vars = {}
    M_vars = {}
    
    for var in model.getVars():
        name = var.varName
        value = var.X
        
        if name.startswith("X_") and value > 0:
            parts = name.split('_')
            if len(parts) >= 5:
                i, j, t, k = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                X_vars[(i, j, t, k)] = value
        
        elif name.startswith("Y_") and value > 0.5:  # Para variables binarias
            parts = name.split('_')
            if len(parts) >= 5:
                i, j, t, k = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                Y_vars[(i, j, t, k)] = value
        
        elif name.startswith("R_") and value > 0.5:
            parts = name.split('_')
            if len(parts) >= 4:
                i, j, k = int(parts[1]), int(parts[2]), int(parts[3])
                R_vars[(i, j, k)] = value
        
        elif name.startswith("Z_plus_") and value > 0.5:
            parts = name.split('_')
            if len(parts) >= 5:
                i, j, k = int(parts[2]), int(parts[3]), int(parts[4])
                Z_plus_vars[(i, j, k)] = value
        
        elif name.startswith("Z_minus_") and value > 0.5:
            parts = name.split('_')
            if len(parts) >= 5:
                i, j, k = int(parts[2]), int(parts[3]), int(parts[4])
                Z_minus_vars[(i, j, k)] = value
        
        elif name.startswith("Z_") and not name.startswith("Z_plus_") and not name.startswith("Z_minus_") and value > 0.5:
            parts = name.split('_')
            if len(parts) >= 4:
                i, j, k = int(parts[1]), int(parts[2]), int(parts[3])
                Z_vars[(i, j, k)] = value
        
        elif name.startswith("M_") and value > 0.5:
            parts = name.split('_')
            if len(parts) >= 5:
                i, j, k, t = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                M_vars[(i, j, k, t)] = value
    
    # Imprimir los resultados
    print("\n=== SISTEMAS DE RIEGO SELECCIONADOS ===")
    for (i, j, k), value in R_vars.items():
        area = a_ij.get((i, j), 0)
        print(f"Área Verde {i}, Planta {j}: Sistema de Riego {k} (Área: {area} m²)")
    
    print("\n=== INSTALACIONES DE NUEVOS SISTEMAS DE RIEGO ===")
    for (i, j, k), value in Z_plus_vars.items():
        area = a_ij.get((i, j), 0)
        print(f"Área Verde {i}, Planta {j}: Instalación de Sistema {k} (Área: {area} m²)")
    
    print("\n=== DESINSTALACIONES DE SISTEMAS DE RIEGO ===")
    for (i, j, k), value in Z_minus_vars.items():
        area = a_ij.get((i, j), 0)
        print(f"Área Verde {i}, Planta {j}: Desinstalación de Sistema {k} (Área: {area} m²)")
    
    print("\n=== MANTENIMIENTOS PROGRAMADOS ===")
    for (i, j, k, t), value in M_vars.items():
        print(f"Día {t}: Mantenimiento en Área Verde {i}, Planta {j}, Sistema {k}")
    
    print("\n=== DESGLOSE DE RIEGO POR DÍA ===")
    dias_set = data['dias_set']  # Obtener conjunto de días
    total_por_dia = {t: 0 for t in dias_set}
    
    for t in dias_set:
        print(f"\nDía {t}:")
        day_total = 0
        
        for (i, j, t_day, k), vol in X_vars.items():
            if t_day == t:
                print(f"  Área Verde {i}, Planta {j}, Sistema {k}: {vol:.2f} litros")
                day_total += vol
        
        total_por_dia[t] = day_total
        print(f"  Total del día: {day_total:.2f} litros")
    
    # Calcular el presupuesto utilizado
    presupuesto_total = data['p']
    presupuesto_usado = 0
    
    for (i, j, k), value in Z_plus_vars.items():
        presupuesto_usado += value * a_ij[(i, j)] * data['g_k'][k]
    
    for (i, j, k), value in Z_minus_vars.items():
        presupuesto_usado += value * a_ij[(i, j)] * data['d_k'][k]
    
    for (i, j, k, t), value in M_vars.items():
        presupuesto_usado += value * a_ij[(i, j)] * data['c_k'][k]
    
    print("\n=== RESUMEN DE PRESUPUESTO ===")
    print(f"Presupuesto disponible: ${presupuesto_total:.2f}")
    print(f"Presupuesto utilizado: ${presupuesto_usado:.2f}")
    print(f"Presupuesto restante: ${presupuesto_total - presupuesto_usado:.2f}")
    
    print(f"\nNota: El modelo se resolvió en {model.Runtime:.2f} segundos.")
    print(f"Estado de la solución: {model.status}")
    print(f"Gap de optimalidad: {model.MIPGap*100:.2f}%")


def main():
    print("Cargando datos...")
    data = cargar_datos()
    
    print("Construyendo modelo...")
    model = construir_modelo(data)
    
    print("Resolviendo modelo...")
    resultado = resolver_modelo(model)
    
    print("Imprimiendo resultados...")
    imprimir_resultados(resultado)


if __name__ == "__main__":
    main()