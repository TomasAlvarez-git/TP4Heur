import random, math
import numpy as np

def geneticoSimple(Pop, cant_generac, Bounds, Enteras, sel, xover, mut, \
                cant_cruzados, cant_mutados, cant_elite, param_sel, f, callback=None):
    """
    Algoritmo Genetico Simple
    Parametros:
    - Pop: list. poblacion inicial (lista de listas).
    - cant_generac: int. Cantidad de iteraciones del algoritmo.
    - Bounds: list: Matriz que indica los valores max y min de cada variable.
    - Enteras: list. Vector booleano que indica si las coordenadas son enteras.
    - cant_cruzados: int. Cantidad de individuos que seran cruzados.
    - cant_mutados: int. Cantidad de soluciones a las cuales se le aplicara mutacion.
    - cant_elite: int. Numero de mejores soluciones que pasan a la siguiente generacion sin cambio alguno.
    - param_sel: float. Parametro usado en la seleccion, por ejemplo el peso que
      se le asigna a la solucion menos probable en el calculo de la probabilidad
      para ruleta. En caso de usar torneo, este es el tamano del torneo.
    - f: function. Funcion de evaluacion. Debe recibir como parametro un vector
      solucion y devolver un numero real.
    - callback: function. Funcion que se ejecuta una vez en cada iteracion (para
      mostrar resultados parciales). Usar None para deshabilitarla.
    Salida
    - soluc: list. Vector solucion que obtuvo fitness mas alto.
    - solF: float. Valor de evaluacion de la mejor solucion.
    - Pop: list. Poblacion final.
    - Fit: list. Fitness de cada individuo de la poblacion final.
    - traceBest: list. Vector con los mejores valores de fitness obtenidos en
      cada generacion. Se puede utilizar para determinar si el algoritmo
      llego a crowding.
    - traceAvg: list. Vector con los valores promedio de fitness
      obtenidos en cada generacion.
    """
    #inicializacion
    sol = []
    solF = None
    cant_soluc = len(Pop)
    Fit = [None] * cant_soluc
    traceBest = []
    traceAvg = []
    bestSols = []

    #verifico los parametros de entrada
    if cant_soluc < 2:
        print('La Poblacion debe tener al menos dos individuos.')
        return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols
    cant_variables = len(Pop[0])
    if len(Bounds) != cant_variables or len(Bounds[0]) != 2:
        print('El tamano de la matriz Bounds es incorrecto.')
        return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols
    if len(Enteras) != cant_variables:
        print('El tamano del vector Enteras es incorrecto.')
        return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols
    if cant_cruzados > cant_soluc:
        print('La cantidad de cruzados debe ser menor que el tamano de la poblacion.')
        return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols
    if cant_mutados > cant_soluc:
        print('La cantidad de mutados debe ser menor que el tamano de la poblacion.')
        return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols
    if cant_elite >= cant_soluc:
        print('El tamano de la elite debe ser menor que el tamano de la poblacion.')
        return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols

    #inicializo variables para reducir tiempo de calculo
    cb_param = None
    p = [i for i in range(cant_soluc)]

    #realizo las evaluaciones de la poblacion inicial.
    for k in range(cant_soluc):
        Pop[k] = [xi if not Enteras[i] else int(xi+0.5) for i, xi in enumerate(Pop[k])]
        Fit[k] = f(Pop[k])
    if callback != None:
        if callback is callback_graficar:
            cb_param = callback(Pop, Fit, Bounds, 0, cant_generac, cb_param, [])
        else:
            cb_param = callback(Pop, Fit, Bounds, 0, cant_generac, cb_param)
    #obtengo el orden de los individuos
    p_orden = sorted(range(len(Fit)), key=lambda k: Fit[k], reverse=True)

    for gen in range(cant_generac):
        elite_positions = []
        #selecciono la elite de la generacion actual (una copia)
        Elite = [[xi for xi in Pop[p_orden[k]]] for k in range(cant_elite)]
        Elite_Fit = [Fit[p_orden[k]] for k in range(cant_elite)]

        #seleccion
        p_sel = sel(Fit, cant_soluc, param_sel)
        random.shuffle(p)
        Pop = [Pop[p_sel[k]] for k in p]
        Fit = [Fit[p_sel[k]] for k in p]

        #cruzamiento
        random.shuffle(p)
        for k in range(0, cant_cruzados-1, 2):
            C1, C2 = xover(Pop[p[k]], Pop[p[k+1]])
            Pop[p[k]] = [xi for xi in C1]
            Pop[p[k+1]] = [xi for xi in C2]
            Fit[p[k]] = None
            Fit[p[k+1]] = None

        #mutacion
        random.shuffle(p)
        for k in range(cant_mutados):
            C = mut(Pop[p[k]], Bounds)
            Pop[p[k]] = [xi for xi in C]
            Fit[p[k]] = None

        #inserto la elite en la nueva poblacion en lugares al azar
        random.shuffle(p)
        for k in range(cant_elite):
            Pop[p[k]] = Elite[k]
            Fit[p[k]] = Elite_Fit[k]
            elite_positions.append(p[k])

        #realizo las evaluaciones de la nueva generacion
        for k in range(cant_soluc):
            if Fit[k] == None:
                Pop[k] = [xi if not Enteras[i] else int(xi+0.5) for i, xi in enumerate(Pop[k])]
                Fit[k] = f(Pop[k]);

        #obtengo el orden de los individuos
        p_orden = sorted(range(len(Fit)), key=lambda k: Fit[k], reverse=True)

        #datos estadisticos
        traceBest.append(Fit[p_orden[0]])
        traceAvg.append(sum(Fit) / len(Fit))
        bestSols.append([xi for xi in Pop[p_orden[0]]])

        #funcion para graficar, mostrar, etc.
        if callback != None:
            if callback is callback_graficar:
                cb_param = callback(
                    Pop, Fit, Bounds, gen + 1, cant_generac, cb_param,
                    elite_positions)
            else:
                cb_param = callback(Pop, Fit, Bounds, gen, cant_generac, cb_param)

    #busco la mejor solucion obtenida
    solF = Fit[p_orden[0]]
    sol = Pop[p_orden[0]]

    return sol, solF, Pop, Fit, traceBest, traceAvg, bestSols

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def graficarEvolucionFitness(best, mean, ax=None):
    """Método para graficar la evolucion del fitness con las iteraciones.
    Parametros
    best, mean : list. Lista con los valores de evaluación de la mejor y peor solución.
    """
    if ax is None:
        _, ax = plt.subplots()
    x = range(len(best));
    ax.plot(x, best, label='Mejor solución');
    ax.plot(x, mean, label='Promedio poblacion');
    ax.set_title('Valores de Evaluación')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Fitness')
    ax.grid()
    ax.legend()

def graficarCaminata(fitness, solutions, bounds, resolution, alpha=0.5, ax=None):
    """Graficar la funcion de evaluacion y las soluciones encontradas.
    Parametros
    fitness : function. Función de evaluación a optimizar
    solutions : list. Lista de soluciones encontradas.
    bounds: list(tuple). Matriz de tamano nx2, donde n es la cantidad de
    variables del problema (cantidad de coordenadas del vector solución).
    resolution : float. Resolución para graficar la función (mayor a 0.1).
    alpha : float. Transparencia para el grafico de la función.
    """
    ranges = []
    for i in range(len(bounds)):
        steps = round((bounds[i][1] - bounds[i][0]) / resolution)
        ranges.append([bounds[i][0] + s*resolution for s in range(steps)])
    if len(bounds) == 1:
        if ax is None:
            _, ax = plt.subplots()
        X = [xi for xi in ranges[0]]
        Y = [fitness(xi) for xi in ranges[0]]
        ax.plot(X, Y, antialiased=False, alpha=alpha)
        ax.set_title('Función de Evaluación')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid()
        x = [s[0] for s in solutions]
        y = [fitness(s) for s in solutions]
        ax.scatter(x, y, c='k', marker='o')
    elif len(bounds) == 2:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        X = []; Y = []; Z = []
        for i, xi in enumerate(ranges[0]):
            X.append([xi for yj in ranges[1]])
            Y.append([yj for yj in ranges[1]])
            Z.append([fitness([xi, yj]) for yj in ranges[1]])
        surf = ax.plot_surface(X, Y, np.array(Z), cmap=cm.coolwarm, linewidth=0,
            antialiased=False, alpha=alpha)
        ax.set_title('Función de Evaluación')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        ax.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        x = [s[0] for s in solutions]
        y = [s[1] for s in solutions]
        z = [fitness(s) for s in solutions]
        ax.scatter(x, y, z, c='k', marker='o')
    else:
        return False
    return True

def callback_print(Pop, Fit, Bounds, gen, cant_generac, params):
    """Mostrar el valor de fitness si se encuentra una mejor solucion en la
    generacion actual"""
    v = max(Fit)
    if gen == 0:
        params = v
    if v >= params:
        p = Fit.index(v)
        print("Gen.{0}: {1} ({2})".format(gen, Pop[p], v))
    return params

def callback_graficar(Pop, Fit, Bounds, gen, cant_generac, params,
                      elite_indices=None):
    if len(Bounds) != 2:
        return params

    if params is None:
        fig, ax = plt.subplots()
        params = {
            "fig": fig,
            "ax": ax,
            "frames": [],
            "index": 0,
        }

        def mostrar_frame(index):
            params["index"] = max(0, min(index, len(params["frames"]) - 1))
            frame = params["frames"][params["index"]]
            scatter.set_offsets(frame["population"])
            elite_scatter.set_offsets(np.array(frame["elite"]).reshape(-1, 2))
            title = "Población inicial" if params["index"] == 0 else \
                "Generación {0}".format(params["index"])
            ax.set_title("{0} - Fitness: {1} (←/→ para recorrer)".format(
                title, frame["fitness"]))
            fig.canvas.draw_idle()

        def al_presionar_tecla(event):
            if event.key == "left":
                mostrar_frame(params["index"] - 1)
            elif event.key == "right":
                mostrar_frame(params["index"] + 1)

        scatter = ax.scatter([], [], c='k', marker='o')
        elite_scatter = ax.scatter([], [], c='blue', marker='o', label='Elite')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid()
        ax.legend()
        fig.canvas.mpl_connect('key_press_event', al_presionar_tecla)
        params["scatter"] = scatter
        params["elite_scatter"] = elite_scatter

    params["frames"].append({
        "population": [list(individuo) for individuo in Pop],
        "elite": [list(Pop[i]) for i in (elite_indices or [])],
        "fitness": max(Fit),
    })
    params["index"] = len(params["frames"]) - 1
    params["scatter"].set_offsets(params["frames"][-1]["population"])
    params["elite_scatter"].set_offsets(
        np.array(params["frames"][-1]["elite"]).reshape(-1, 2))
    title = "Población inicial" if params["index"] == 0 else \
        "Generación {0}".format(params["index"])
    params["ax"].set_title("{0} - Fitness: {1} (←/→ para recorrer)".format(
        title, max(Fit)))
    params["fig"].canvas.draw_idle()
    if params["index"] == 0:
        plt.show(block=False)
    plt.pause(0.001)
    return params

def init_pop(cant_individuos, Bounds):
    return [[random.uniform(B[0],B[1]) for B in Bounds] for i in range(cant_individuos)]

def acumulada(prob):
    """Acumulada clasica de una distribucion de probabilidad."""
    acum, total = [], 0.0
    for p in prob:
        total += p
        acum.append(total)
    acum[-1] = 1.0
    return acum
def normalizar(F, eps):
    """Desplaza los fitness a positivos y los convierte en probabilidades."""
    m = min(F)
    f_a = [fi - m + eps for fi in F]
    s = sum(f_a)
    return [fi / s for fi in f_a]

def sel_ruleta(F, cant_selectos, eps):
    """ Operador de selección por torneos.
    Parametros:
    F: list. Lista de valores de fitness de cada individuo de la población.
    cant_selectos: int. Cantidad de individuos a seleccionar.
    eps: int. valor para normalizar el fitness.
    S: list. Lista con las posiciones de los individuos seleccionados.
    """

    acum = acumulada(normalizar(F, eps))
    paso = 1.0 / cant_selectos
    inicio = random.random() * paso          # único número aleatorio
    tiros = [inicio + k * paso for k in range(cant_selectos)]
    S, i = [], 0
    for puntero in tiros:
        while acum[i] <= puntero:
            i += 1
        S.append(i)

    return S

def xov_uniform(P1, P2):
    """Operador Uniform Crossover para Algoritmos Geneticos.
    Parametros:
    P1: list. Lista correspondiente a uno de los individuos padres a cruzar.
    P2: list. Lista correspondiente a uno de los individuos padres a cruzar.
    """

    n = len(P1)

    mascara = [random.randint(0, 1) for i in range(n)]
    print(mascara)

    C1 = [P1[i] if mascara[i] == 1 else P2[i] for i in range(n)]
    C2 = [P2[i] if mascara[i] == 1 else P1[i] for i in range(n)]

    return C1, C2

def xov_arith(P1, P2, explore=0):
    """Operador Arith Crossover para Algoritmos Geneticos.
    Parametros:
    P1: list: Lista correspondiente a uno de los individuos padres a cruzar.
    P2: list: Lista correspondiente a uno de los individuos padres a cruzar.
    explore: float. Constante real que indica cuanto me puedo exceder del
    hipercubo formado por las 2 soluciones. Se puede asignar por ejemplo, 0.25.
    """
    a = random.random() * (1 + explore)
    C1 = [P1[i] * a + P2[i] * (1-a) for i in range(len(P1))]
    C2 = [P1[i] * (1-a) + P2[i] * a for i in range(len(P1))]
    return C1, C2

def mut_boundary(P, Bounds):
    """Step Mutation
    Parámetros
    P: list. Vector correspondiente a un individuo.
    bounds: list. Matriz que indica los valores maximo y minimo de cada coordenada.
    """
    C = P.copy()
    cut = random.randint(0,len(P)-1)
    C[cut] = Bounds[cut][random.randint(0,1)]
    return C

def mut_step(P, Bounds):
    """Step Mutation
    Parámetros
    P: list. Vector correspondiente a un individuo.
    bounds: list. Matriz que indica los valores maximo y minimo de cada coordenada.
    """
    C=mut_boundary(P, Bounds)

    Dif = list(set(P) - set(C))
    print(Dif)

    if not Dif:
        return C

    index=P.index(Dif[0])
    C[index]=abs(round((Dif[0]-C[index])*random.random()-Dif[0],2))

    return C


def funcion_escalon(x_vars):
    return sum([int(xi) for xi in x_vars])


if __name__ == "__main__":
    Bounds = [[-5.12, 5.12],[-5.12, 5.12]]
    Enteras = [False]*len(Bounds)
    f = funcion_escalon
    cant_individuos = 20
    cant_generac = 25
    cant_elite = round(cant_individuos * 0.2)
    sel = sel_ruleta
    eps = 0.08
    xover = xov_uniform
    cant_cruzados = round(0.8 * cant_individuos)
    mut = mut_step
    cant_mutados = round(0.2 * cant_individuos)

    #usar callback = None para que no grafique en cada iteracion y vaya mas rapido
    callback = callback_graficar

    Pop = init_pop(cant_individuos, Bounds)
    sol, solF, Pop, Fit, traceBest, traceAvg, bestSols = \
        geneticoSimple(Pop, cant_generac, Bounds, Enteras, sel, xover, mut, \
            cant_cruzados, cant_mutados, cant_elite, eps, f, callback)

    print("Mejor solución: {0} (Fitness: {1})".format(sol, solF))

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    if len(Bounds) == 2:
        ax_caminata = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax_caminata = fig.add_subplot(1, 2, 1)
    ax_fitness = fig.add_subplot(1, 2, 2)

    graficarCaminata(f, bestSols, Bounds, 0.2, alpha=0.2, ax=ax_caminata)
    graficarEvolucionFitness(traceBest, traceAvg, ax=ax_fitness)
    plt.show()