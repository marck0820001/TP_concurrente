int gradW[5] = 0;  // Simulación de gradientes para 5 pesos
int gradB = 0;     // Gradiente del sesgo
byte mutex = 0;    // 0 = libre, 1 = ocupao


proctype Worker(byte id) {
    int localGradW[5];
    int localGradB;

    // Simulamos el cálculo de gradientes locales
    int i;
    do
    :: i < 5 ->
        localGradW[i] = 1; // simulamos una contribución
        i++
    :: else -> break
    od;
    localGradB = 1;

    // Sección crítica para acumular gradientes
    atomic {
        assert(mutex == 0); // Exclusión mutua
        mutex = 1;

        i = 0;
        do
        :: i < 5 ->
            gradW[i] = gradW[i] + localGradW[i];
            i++
        :: else -> break
        od;
        gradB = gradB + localGradB;

        assert(mutex == 1); // Verifica que el mutex esté bloqueado durante la sección crítica

        mutex = 0;

         assert(mutex == 0); // Verifica que el mutex esté desbloqueado después de la sección crítica
    
    }
}

init {
    byte i = 0;
    do
    :: i < 4 ->     // simulamos 4 workers (como 4 goroutines)
        run Worker(i);
        i++
    :: else -> break
    od;
}
