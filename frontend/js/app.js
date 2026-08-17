document.addEventListener('DOMContentLoaded', () => {
    const pantalla1 = document.getElementById('pantalla-1');
    const pantalla2 = document.getElementById('pantalla-2');
    const pantalla3 = document.getElementById('pantalla-3');

    const audioInput = document.getElementById('audioInput');
    const labelArchivo = document.getElementById('labelArchivo');
    const btnTranscribir = document.getElementById('btnTranscribir');
    const textoResultado = document.getElementById('textoResultado');

    const btnDescargarTXT = document.getElementById('btnDescargarTXT');
    const btnDescargarPDF = document.getElementById('btnDescargarPDF');
    const btnNuevoAudio = document.getElementById('btnNuevoAudio');

    let transcripcionFinal = "";

    const modalAlerta = document.getElementById('modalAlerta');
    const textoAlerta = document.getElementById('textoAlerta');
    const btnCerrarAlerta = document.getElementById('btnCerrarAlerta');

    function mostrarAlerta(mensaje) {
        textoAlerta.textContent = mensaje;
        modalAlerta.classList.add('mostrar');
    }

    btnCerrarAlerta.addEventListener('click', () => {
        modalAlerta.classList.remove('mostrar');
    });

    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            labelArchivo.textContent = file.name;
            btnTranscribir.disabled = false;
        } else {
            labelArchivo.innerHTML = '<span class="icono-archivo">♫</span> Elegir Archivo (.WAV o .MP3)';
            btnTranscribir.disabled = true;
        }
    });

    btnTranscribir.addEventListener('click', async () => {
        const file = audioInput.files[0];
        if (!file) return;

        textoResultado.textContent = "";
        transcripcionFinal = "";

        cambiarPantalla(pantalla1, pantalla2);

        const formData = new FormData();
        formData.append("audio", file);

        try {
            const response = await fetch("http://localhost:8000/transcribir", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                if (response.status === 400) {
                    throw new Error("FORMATO_INVALIDO");
                }
                if (response.status === 500) {
                    throw new Error("AUDIO_CORRUPTO");
                }
                throw new Error("Error en la transcripción");
            }

            const data = await response.json();

            if (data.transcripcion.trim() === "") {
                transcripcionFinal = "[Ruido detectado. El modelo no encontró palabras claras]";
            } else {
                transcripcionFinal = data.transcripcion;
            }

            textoResultado.textContent = transcripcionFinal;

            cambiarPantalla(pantalla2, pantalla3);

        } catch (error) {
            if (error.message === "AUDIO_CORRUPTO") {
                mostrarAlerta("El archivo de audio está corrupto o tiene un formato interno no soportado. Por favor, intenta subir un archivo diferente.");
            } else if (error.message === "FORMATO_INVALIDO") {
                mostrarAlerta("Formato no permitido. Por favor, asegúrate de subir únicamente archivos con extensión .wav o .mp3.");
            } else {
                mostrarAlerta("Error conectando con el servidor. Verifica que main.py esté corriendo.");
            }
            audioInput.value = "";
            labelArchivo.innerHTML = '<span class="icono-archivo">♫</span> Elegir Archivo (.WAV o .MP3)';
            btnTranscribir.disabled = true;
            textoResultado.textContent = "Error al procesar el audio.";
            cambiarPantalla(pantalla2, pantalla1); 
        }
    });

    btnDescargarTXT.addEventListener('click', () => {
        const blob = new Blob([transcripcionFinal], { type: "text/plain" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "transcripcion_kichwa.txt";
        a.click();
        window.URL.revokeObjectURL(url);
    });

    btnDescargarPDF.addEventListener('click', () => {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        doc.setFontSize(16);
        doc.text("Transcripción Kichwa", 20, 20);

        doc.setFontSize(12);
        const lineas = doc.splitTextToSize(transcripcionFinal, 170);
        doc.text(lineas, 20, 30);

        doc.save("transcripcion_kichwa.pdf");
    });

    btnNuevoAudio.addEventListener('click', () => {
        audioInput.value = "";
        labelArchivo.innerHTML = '<span class="icono-archivo">♫</span> Elegir Archivo (.WAV o .MP3)';
        btnTranscribir.disabled = true;
        cambiarPantalla(pantalla3, pantalla1);
    });

    function cambiarPantalla(pantallaActual, pantallaNueva) {
        pantallaActual.classList.remove('activa');
        pantallaNueva.classList.add('activa');
    }
});