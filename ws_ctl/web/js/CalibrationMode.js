import Socket from "./Socket.js";

export default class CalibrationMode {
    #enabled = false;
    #calibCtlSocket;
    #cvCtlSocket;

    constructor(cvCtlSocket) {
        this.#cvCtlSocket = cvCtlSocket;

        document.addEventListener('change', (event) => {
            if (event.target.id === 'calibration-switch') {
                this.toggleState(event.target.checked);
            }
        });

        this.#calibCtlSocket = new Socket("/calib_ctl", (data) => {
            document.addEventListener('socket.calibCtl.onmessage', new CustomEvent({detail: {data}}));
        });

        document.addEventListener('socket.cvCtl.onmessage', (event) => {
            const message = JSON.parse(event.detail);
            if (message[0] === 'STARTED_CALIBRATION') {
                this.#calibCtlSocket.connect();
                this.#toggleElementsState(true);
                console.log(message);
            } else if (message[0] === 'STOPPED_CALIBRATION') {
                this.#calibCtlSocket.disconnect();
                this.#toggleElementsState(false);
                console.log(message);
            }
        });
    }

    toggleState(toState) {
        if (!!toState) {
            this.#cvCtlSocket.sendMessage(JSON.stringify(['START_CALIBRATION']))
        } else {
            this.#cvCtlSocket.sendMessage(JSON.stringify(['STOP_CALIBRATION']))
        }
        this.#enabled = toState;
    }

    #toggleElementsState(toState) {
        const elements = document.getElementsByClassName('calibration-mode');
        if (toState) {
            for (const e of elements) {
                e.classList.remove('is-hidden');
            }
        } else {
            for (const e of elements) {
                e.classList.add('is-hidden');
            }
        }
    }
}