document.addEventListener('DOMContentLoaded', () => {
    const pantalla1 = document.getElementById('pantalla-1');
    const pantalla2 = document.getElementById('pantalla-2');
    const pantalla3 = document.getElementById('pantalla-3');

    const audioInput = document.getElementById('audioInput');
    const labelArchivo = document.getElementById('labelArchivo');
    const btnTranscribir = document.getElementById('btnTranscribir');
    const textoResultado = document.getElementById('textoResultado');
    const btnGrabar = document.getElementById('btnGrabar');
    const textoBtnGrabar = document.getElementById('textoBtnGrabar');
    const estadoGrabacion = document.getElementById('estadoGrabacion');
    const audioPreview = document.getElementById('audioPreview');

    const btnDescargarTXT = document.getElementById('btnDescargarTXT');
    const btnDescargarPDF = document.getElementById('btnDescargarPDF');
    const btnNuevoAudio = document.getElementById('btnNuevoAudio');

    let transcripcionFinal = "";
    let archivoAudio = null;
    let audioContext = null;
    let mediaStream = null;
    let sourceNode = null;
    let processorNode = null;
    let fragmentosAudio = [];
    let sampleRate = 44100;
    let inicioGrabacion = 0;
    let intervaloGrabacion = null;
    let previewUrl = null;

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
            limpiarGrabacion();
            archivoAudio = file;
            labelArchivo.textContent = file.name;
            btnTranscribir.disabled = false;
        } else {
            archivoAudio = null;
            labelArchivo.innerHTML = '<span class="icono-archivo">♫</span> Elegir Archivo (.WAV o .MP3)';
            btnTranscribir.disabled = true;
        }
    });

    btnGrabar.addEventListener('click', async () => {
        if (mediaStream) detenerGrabacion();
        else await iniciarGrabacion();
    });

    async function iniciarGrabacion() {
        if (!navigator.mediaDevices?.getUserMedia) {
            mostrarAlerta("Tu navegador no permite grabar audio. Usa una versión reciente de Chrome, Edge o Firefox.");
            return;
        }
        try {
            btnGrabar.disabled = true;
            limpiarGrabacion();
            audioInput.value = "";
            labelArchivo.innerHTML = '<span class="icono-archivo">♫</span> Elegir Archivo (.WAV o .MP3)';
            archivoAudio = null;
            btnTranscribir.disabled = true;

            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            btnGrabar.disabled = false;
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            sampleRate = audioContext.sampleRate;
            fragmentosAudio = [];
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            processorNode.onaudioprocess = (event) => {
                fragmentosAudio.push(new Float32Array(event.inputBuffer.getChannelData(0)));
            };
            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);

            inicioGrabacion = Date.now();
            btnGrabar.classList.add('grabando');
            textoBtnGrabar.textContent = "Detener grabación";
            actualizarTiempoGrabacion();
            intervaloGrabacion = setInterval(actualizarTiempoGrabacion, 1000);
        } catch (error) {
            btnGrabar.disabled = false;
            limpiarRecursosGrabacion();
            mostrarAlerta("No se pudo acceder al micrófono. Revisa el permiso del navegador e inténtalo nuevamente.");
        }
    }

    function detenerGrabacion() {
        limpiarRecursosGrabacion();
        clearInterval(intervaloGrabacion);
        intervaloGrabacion = null;
        btnGrabar.classList.remove('grabando');
        textoBtnGrabar.textContent = "Grabar de nuevo";

        const muestras = unirFragmentos(fragmentosAudio);
        if (muestras.length === 0) {
            estadoGrabacion.textContent = "No se capturó audio";
            return;
        }
        const wavBlob = crearWav(muestras, sampleRate);
        archivoAudio = new File([wavBlob], `grabacion_${Date.now()}.wav`, { type: "audio/wav" });
        previewUrl = URL.createObjectURL(wavBlob);
        audioPreview.src = previewUrl;
        audioPreview.hidden = false;
        estadoGrabacion.textContent = `Grabación lista · ${formatearTiempo(Date.now() - inicioGrabacion)}`;
        btnTranscribir.disabled = false;
    }

    function actualizarTiempoGrabacion() {
        estadoGrabacion.textContent = `Grabando… ${formatearTiempo(Date.now() - inicioGrabacion)}`;
    }

    function formatearTiempo(milisegundos) {
        const total = Math.floor(milisegundos / 1000);
        return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
    }

    function unirFragmentos(fragmentos) {
        const longitud = fragmentos.reduce((total, fragmento) => total + fragmento.length, 0);
        const resultado = new Float32Array(longitud);
        let offset = 0;
        fragmentos.forEach((fragmento) => { resultado.set(fragmento, offset); offset += fragmento.length; });
        return resultado;
    }

    function crearWav(muestras, frecuencia) {
        const buffer = new ArrayBuffer(44 + muestras.length * 2);
        const vista = new DataView(buffer);
        const escribir = (offset, texto) => {
            for (let i = 0; i < texto.length; i++) vista.setUint8(offset + i, texto.charCodeAt(i));
        };
        escribir(0, 'RIFF'); vista.setUint32(4, 36 + muestras.length * 2, true);
        escribir(8, 'WAVE'); escribir(12, 'fmt '); vista.setUint32(16, 16, true);
        vista.setUint16(20, 1, true); vista.setUint16(22, 1, true);
        vista.setUint32(24, frecuencia, true); vista.setUint32(28, frecuencia * 2, true);
        vista.setUint16(32, 2, true); vista.setUint16(34, 16, true);
        escribir(36, 'data'); vista.setUint32(40, muestras.length * 2, true);
        let offset = 44;
        muestras.forEach((muestra) => {
            const limitada = Math.max(-1, Math.min(1, muestra));
            vista.setInt16(offset, limitada < 0 ? limitada * 0x8000 : limitada * 0x7fff, true);
            offset += 2;
        });
        return new Blob([vista], { type: 'audio/wav' });
    }

    function limpiarRecursosGrabacion() {
        if (processorNode) processorNode.disconnect();
        if (sourceNode) sourceNode.disconnect();
        if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
        if (audioContext && audioContext.state !== 'closed') audioContext.close();
        processorNode = null; sourceNode = null; mediaStream = null; audioContext = null;
    }

    function limpiarGrabacion() {
        limpiarRecursosGrabacion();
        clearInterval(intervaloGrabacion);
        intervaloGrabacion = null;
        fragmentosAudio = [];
        btnGrabar.classList.remove('grabando');
        textoBtnGrabar.textContent = "Iniciar grabación";
        estadoGrabacion.textContent = "Usa el micrófono de tu dispositivo";
        audioPreview.hidden = true;
        audioPreview.removeAttribute('src');
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    btnTranscribir.addEventListener('click', async () => {
        const file = archivoAudio;
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
            archivoAudio = null;
            limpiarGrabacion();
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
        archivoAudio = null;
        limpiarGrabacion();
        labelArchivo.innerHTML = '<span class="icono-archivo">♫</span> Elegir Archivo (.WAV o .MP3)';
        btnTranscribir.disabled = true;
        cambiarPantalla(pantalla3, pantalla1);
    });

    function cambiarPantalla(pantallaActual, pantallaNueva) {
        pantallaActual.classList.remove('activa');
        pantallaNueva.classList.add('activa');
    }
});
