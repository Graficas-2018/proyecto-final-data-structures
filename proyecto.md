# Definicion de Proyecto
Jose Enrique Estremadoyro Fort
Para computacion multinucleo

## Objetivo
Crear con multinucleo una estructura de datos que utilice algoritmos que aprovechen el multinucleo de forma que se libere esta estrcutura como open source.

## Propuesta

Una estructura que haga ordenamiento, maximo y minimo utilizando como base el procesamiento multinucleo y haciendo lo mas paralelizable sus procesos. Una idea para el ordenamiento seria hacerlo totalmente multinucleo en el sentido de que se paraleliza la comparacion de un numero con todos tratando de sacar su posicion en varios nucleos es decir: Se compara un numero con todos y su posicion se define por el numero de numeros menores que tiene. Normalmente algo asi seria muy caro n^2, pero cuando esto se hace simultaneamente se podria considerar de orden 1. Simplemente en el primer paso se hacen todas las pruebas y en el segundo se suman los resultados por numero. El diseño de una estructura de datos con algoritmos de ordenamiento y otras cosas, es la propuesta a seguir e implementarla en cuda con ejemplos y tests automatizados, comparandola con una similar a fin de liberarlo como codigo abierto.
