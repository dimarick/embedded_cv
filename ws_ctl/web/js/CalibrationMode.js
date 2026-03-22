import Socket from "./Socket.js";

export default class CalibrationMode {
    #enabled = false;
    #calibCtlSocket;
    #cvCtlSocket;

    constructor(cvCtlSocket, calibCtlSocket) {
        this.#cvCtlSocket = cvCtlSocket;
        this.#calibCtlSocket = calibCtlSocket;

        this.toggleState(document.getElementById('calibration-switch').checked);

        document.addEventListener('change', (event) => {
            if (event.target.id === 'calibration-switch') {
                this.toggleState(event.target.checked);
            }
        });

        document.addEventListener('socket.cvCtl.onmessage', (event) => {
            const message = JSON.parse(event.detail);
            if (message[0] === 'SUSPENDED_CV') {
                this.#toggleUiState(true);
                console.log(message);
            } else if (message[0] === 'RESUMED_CV') {
                this.#toggleUiState(false);
                console.log(message);
            }
        });

        document.addEventListener('socket.calibCtl.onmessage', (event) => {
            const message = JSON.parse(event.detail);
            if (message[0] === 'STARTED_CALIBRATION') {
                this.#cvCtlSocket.sendMessage(JSON.stringify(['SUSPEND_CV']));
                console.log(message);
            } else if (message[0] === 'STOPPED_CALIBRATION') {
                this.#cvCtlSocket.sendMessage(JSON.stringify(['RESUME_CV']));
                console.log(message);
            }
        });
    }

    toggleState(toState) {
        if (!!toState) {
            this.#calibCtlSocket.sendMessage(JSON.stringify(['START_CALIBRATION']))
        } else {
            this.#calibCtlSocket.sendMessage(JSON.stringify(['STOP_CALIBRATION']))
        }
        this.#enabled = toState;
    }

    #toggleUiState(toState) {
        const elementsEnabled = document.getElementsByClassName('calibration-mode-enabled');
        const elementsDisabled = document.getElementsByClassName('calibration-mode-disabled');
        if (toState) {
            for (const e of elementsEnabled) {
                e.classList.remove('is-hidden');
            }
            for (const e of elementsDisabled) {
                e.classList.add('is-hidden');
            }
        } else {
            for (const e of elementsEnabled) {
                e.classList.add('is-hidden');
            }
            for (const e of elementsDisabled) {
                e.classList.remove('is-hidden');
            }
        }

        document.getElementById('calibration-switch').checked = toState;
    }
}