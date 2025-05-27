    \section{Modelo del Problema}
        \subsection{Conjuntos}
            $T$ : Conjunto de días $t \in \{$1,...,T$\} $ \\
            $I$ : Conjunto de áreas verdes urbanas $i \in \{$1,...,I$\} $ \\
            $J$ : Conjunto de plantas $j \in \{$1,...,J$\} $ \\
            $K$ : Conjunto de sistemas de riego $k \in \{$1,...,K$\} $ \\
        \subsection{Parámetros}
            $a_{ij}$ : área en $m^{2}$ de planta $j$ en el área verde $i$. \\
            $b_{j}$ : volumen de agua en litros $l$ que requiere la planta $j$ por $m^{2}$ en un día de riego. \\
            $f_{max\;j}$ : frecuencia máxima de riego en días que soporta una planta $j$ \\
            $f_{min\;j}$ : frecuencia mínima de riego en días que soporta una planta $j$ \\
            $e_{kj}$ : eficiencia del sistema de riego $k$ para la planta $j$. Corresponde a un valor $\in \; [0,1]$. \\
            $w_{tij}$ : volumen de agua en litros $l$ de riego natural (lluvia y/o rocío) en el día $t$ para la planta $j$ en el área verde $i$.\\
            $v_{ijt}$ :
            $\begin{cases}
            1 & \text{si se considera regada la planta $j$ del área verde $i$ el día $t$. Es decir, si $\frac{w_{tij}}{b_{j}} \geq 1$.}\\
            0 & \text{e.o.c.} \\
            \end{cases}$ \\
            $q_t$ : volumen de agua en litros $l$ disponibles para riego durante el día $t$ \\
            $s_{ijk}$ :
            $\begin{cases}
            1 & \text{si existe previamente el sistema de riego $k$ para la planta $j$ en el área verde $i$.}\\
            0 & \text{e.o.c.} \\
            \end{cases}$ \\
            $g_{k}$ : costo instalación de sistema de riego $k$ por $m^{2}$.\\
            $d_{k}$ : costo desinstalación de sistema de riego $k$ por $m^{2}$.\\
            $c_{k}$ : costo mantenimiento de sistema de riego $k$ por $m^{2}$.\\
            $l_{k}$ : frecuencia máxima de mantenimiento de sistema de riego $k$ en días.\\
            $p$ : presupuesto municipal para desinstalación, instalación y mantenimiento de sistemas de riego.\\
        \subsection{Variables}
        $X_{ijtk}$ : volumen de agua en litros $l$ destinada a la planta $j$ en el área verde $i$ el día $t$ mediante el sistema de riego $k$.
        \[
            \begin{array}{ll}
            Y_{ijtk} : &
            \begin{cases}
            1 & \text{si se riega la planta $j$ del área verde $i$ el día $t$ mediante el sistema de riego $k$} \\
            0 & \text{e.o.c.}
            \end{cases} \\
            R_{ijk} : &
            \begin{cases}
            1 & \text{si se utiliza el sistema de riego $k$ en el área verde $i$ para la planta $j$} \\
            0 & \text{e.o.c.}
            \end{cases} \\
            Z_{ijk}^{-} : &
            \begin{cases}
            1 & \text{si se desinstala el sistema de riego $k$ en el área verde $i$ para la planta $j$} \\
            0 & \text{e.o.c.}
            \end{cases} \\
            Z_{ijk} : &
            \begin{cases}
            1 & \text{si se mantiene el sistema de riego $k$ en el área verde $i$ para la planta $j$} \\
            0 & \text{e.o.c.}
            \end{cases} \\
            Z_{ijk}^{+} : &
            \begin{cases}
            1 & \text{si se instala el sistema de riego $k$ en el área verde $i$ para la planta $j$} \\
            0 & \text{e.o.c.}
            \end{cases} \\
            M_{ijkt} : &
            \begin{cases}
            1 & \text{si se realiza el mantenimiento del sistema de riego $k$ en el área verde $i$ para la planta $j$ el día $t$} \\
            0 & \text{e.o.c.}
            \end{cases} \\
            \end{array}
            \]


        \subsection{Función Objetivo}
            \[ \text{min}
            \sum_{t \; \in \; T} \sum_{k \; \in \; K} \sum_{i \; \in \; I} \sum_{j \; \in \; J} X_{ijtk}
            \]
        \subsection{Restricciones}
            \begin{itemize}
                \item[1.] Cumplir con el requerimiento hídrico de la planta $j$.
                \[
                b_{j} \; a_{ij} \; Y_{ijtk} \leq w_{ijtk} + X_{ijtk} \; e_{jk} \quad \forall \; i \in I , \forall \; j \in J , \forall \; t \in T , \forall \; k \in K
                \]
                \item[2.] Cumplir con la frecuencia mínima de la planta $j$.
                \[ M(1 - Y_{ijkt}) \geq
                \sum_{t' = t - f_{min \; j} +1}^{ t - 1} (Y_{ijkt'} \; + \; v_{ijt'}) \quad \forall \; i \in I , \forall \; j \in J , \forall \; t \in \{f_{min\; j}, ..., T \} , \forall \; k \in K
                \]
                \item[3.] Cumplir con la frecuencia máxima de la planta $j$.
                \[ 1 - Y_{ijkt} \leq
                \sum_{t' = t - f_{max \; j} +1}^{ t - 1} (Y_{ijkt'} \; + \; v_{ijt'}) \quad \forall \; i \in I , \forall \; j \in J , \forall \; t \in \{f_{max\; j}, ..., T \} , \forall \; k \in K
                \]
                \item[4.] Si una planta $j$ se considera regada, entonces no se riega.
                \[
                v_{ijt} + Y_{ijtk} \leq 1 \quad \forall \; i \in I , \forall \; j \in J , \forall \; t \in T , \forall \; k \in K
                \]
                \item[5.] Si no se riega una planta $j$ entonces el agua destinada a riego es 0.
                \[
                M \; Y_{ijtk} \geq X_{ijtk} \quad \forall \; i \in I , \forall \; j \in J , \forall \; t \in T , \forall \; k \in K
                \]
                 \item[6.] Escoger el sistema de riego $k$ con una eficiencia mínima de 0.8 con $j$.
                \[
                e_{jk} \geq 0.8 \; R_{ijk} \quad \forall \; i \in I , \forall \; j \in J , \forall \; k \in K
                \]
                \item[7.] Si no se encuentra implementado el sistema de riego $k$, entonces no se riega mediante tal sistema.
                \[R_{ijk} \geq Y_{ijtk} \quad \forall \; i \in I , \forall \; j \in J , \forall \; t \in T , \forall \; k \in K
                \]
                \item[8.] No sobrepasar el volumen de agua disponible para riegos.
                \[
                \sum_{k \; \in \; K} \sum_{i \; \in \; I} \sum_{j \; \in \; J} X_{ijtk} \leq q_t \quad \forall \; t \in T
                \]
                \item[9.] Se debe escoger uno y solo un sistema de riego por planta $j$ en el área verde $i$.
                \[
                \sum_{k \; \in \; K} R_{ijk} = 1 \quad \forall \; i \in I , \forall \; j \in J
                \]
                \item[10.] Solo se puede mantener un sistema de riego si estaba previamente instalado.
                \[
                Z_{ijk} \leq s_{ijk} \quad \forall i \in I,\; \forall j \in J,\; \forall k \in K
                \]
                \item[11.] No se puede instalar y mantener el mismo sistema de riego.
                \[
                Z_{ijk} + Z_{ijk}^{+} \leq 1 \quad \forall i \in I,\; \forall j \in J,\; \forall k \in K
                \]
                \item[12.] No se puede desinstalar y mantener el mismo sistema de riego.
                \[
                Z_{ijk} + Z_{ijk}^{-} \leq 1 \quad \forall i \in I,\; \forall j \in J,\; \forall k \in K
                \]
                \item[13.] No se puede desinstalar e instalar el mismo sistema de riego.
                \[
                Z_{ijk}^{-} + Z_{ijk}^{+} \leq 1 \quad \forall i \in I,\; \forall j \in J,\; \forall k \in K
                \]
                \item[14.] Si es que se decide instalar un sistema de riego $k$, debe ser solo uno.
                \[
                \sum_{k \; \in \; K}  Z_{ijk}^{+} \leq 1 \quad \forall \; i \in I , \forall \; j \in J
                \]
                \item[15.] Si es que se decide desinstalar un sistema de riego $k$, debe ser solo uno.
                \[
                \sum_{k \; \in \; K}  Z_{ijk}^{-} \leq 1 \quad \forall \; i \in I , \forall \; j \in J
                \]
                \item[16.] Si se decide mantener un riego $k$, entonces no se desinstala nada.
                \[
                \sum_{k' \; \in \; K}  Z_{ijk'}^{-} = 1 - Z_{ijk} \quad \forall \; i \in I , \forall \; j \in J, \; \forall \; k \in K
                \]
                \item[17.] Si se decide mantener un riego $k$, entonces no se instala nada.
                \[
                \sum_{k' \; \in \; K}  Z_{ijk'}^{+} = 1 - Z_{ijk} \quad \forall \; i \in I , \forall \; j \in J, \; \forall \; k \in K
                \]
                \item[18.] Si se decide mantener un riego $k$, entonces es el riego escogido.
                \[
                 Z_{ijk} = R_{ijk}\quad \forall \; i \in I , \forall \; j \in J, \; \forall \; k \in K
                \]
                \item[19.] Si se decide instalar un riego $k$, entonces es el riego escogido.
                \[
                 Z_{ijk}^{+} = R_{ijk}\quad \forall \; i \in I , \forall \; j \in J, \; \forall \; k \in K
                \]
                \item[20.] Se debe respetar la frecuencia máxima de mantenimiento del sistema de riego $k$. Es decir, se debe hacer un mantenimiento como máximo cada $l_k$ días de uso efectivo.
                \[
                \sum_{t' = 1}^{t - 1} Y_{ijkt'} - l_k \cdot \sum_{t' = 1}^{t - 1} M_{ijkt'} \leq (l_k - 1) + M_{ijkt} \quad \forall \; i \in I ,\; \forall \; j \in J ,\; \forall \; k \in K ,\; \forall \; t \in T
                \]
                \item[21.] Solo se puede realizar mantenimiento a los sistemas de riego $k$ implementados.
                \[
                M_{ijkt} \leq R_{ijk} \quad \forall \; i \in I,\; \forall \; j \in J,\; \forall \; k \in K,\; \forall \; t \in T
                \]

                \item[22.] Se debe respetar el presupuesto municipal $p$ para la desinstalación, instalación y mantenimiento de los sistemas de riego $k$.
                \[
                \sum_{k \in K} \sum_{i \in I} \sum_{j \in J} \left(
                Z_{ijk}^{-} a_{ij} d_k +
                Z_{ijk}^{+} a_{ij} g_k +
                \sum_{t \in T} M_{ijkt} a_{ij} c_k
                \right)
                \leq p
                \]

                \item[23.] Naturaleza de las variables.
                \[
                Y_{ijtk},\; R_{ijk},\; Z_{ijk}^{-},\; Z_{ijk}^{+},\; Z_{ijk},\; M_{ijkt} \in \{0,1\} \quad \forall \; i \in I,\; \forall \; j \in J,\; \forall \; k \in K,\; \forall \; t \in T
                \]
                \[
                X_{ijtk} \geq 0 \quad \forall \; i \in I,\; \forall \; j \in J,\; \forall \; k \in K,\; \forall \; t \in T
                \]
            \end{itemize}
