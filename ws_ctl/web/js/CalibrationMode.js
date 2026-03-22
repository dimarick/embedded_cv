import Socket from "./Socket.js";

export default class CalibrationMode {
    #enabled = false;
    #started = false;
    #calibCtlSocket;
    #cvCtlSocket;

    constructor(cvCtlSocket, calibCtlSocket) {
        this.#cvCtlSocket = cvCtlSocket;
        this.#calibCtlSocket = calibCtlSocket;

        this.#toggleState(document.getElementById('calibration-switch').checked);

        document.addEventListener('change', (event) => {
            if (event.target.id === 'calibration-switch') {
                this.#toggleState(event.target.checked);
            }
        });

        document.addEventListener('click', async (event) => {
            if (event.target.id === 'calibrate-start') {
                await this.#start();
                this.#started = true;
            } else if (event.target.id === 'calibrate-stop') {
                await this.#stop();
                this.#started = false;
            } else if (event.target.id === 'calibrate-reset') {
                await this.#reset();
            } else if (event.target.id === 'calibrate-save') {
                await this.#stop();
                this.#started = false;
                await this.#save();
            } else {
                return;
            }
            this.#toggleUiState();
        });

        document.addEventListener('socket.cvCtl.onmessage', (event) => {
            const message = JSON.parse(event.detail);
            if (message[0] === 'SUSPENDED_CV') {
                this.#enabled = true;
                this.#started = false;
                this.#toggleUiState();
                console.log(message);
            } else if (message[0] === 'RESUMED_CV') {
                this.#enabled = false;
                this.#started = false;
                this.#toggleUiState();
                console.log(message);
            }
        });

        document.addEventListener('socket.calibCtl.onmessage', (event) => {
            const message = JSON.parse(event.detail);
            if (message[0] === 'ENABLED_CALIBRATION') {
                this.#cvCtlSocket.sendMessage(JSON.stringify(['SUSPEND_CV']));
                console.log(message);
            } else if (message[0] === 'DISABLED_CALIBRATION') {
                this.#cvCtlSocket.sendMessage(JSON.stringify(['RESUME_CV']));
                console.log(message);
            }
        });
    }

    #toggleState(toState) {
        if (!!toState) {
            this.#calibCtlSocket.sendMessage(JSON.stringify(['ENABLE_CALIBRATION']))
        } else {
            this.#calibCtlSocket.sendMessage(JSON.stringify(['DISABLE_CALIBRATION']))
        }
        this.#enabled = toState;
    }

    #toggleUiState() {
        const elementsEnabled = document.getElementsByClassName('calibration-mode-enabled');
        const elementsDisabled = document.getElementsByClassName('calibration-mode-disabled');
        const elementsStarted = document.getElementsByClassName('calibration-mode-started');
        const elementsStopped = document.getElementsByClassName('calibration-mode-stopped');
        if (this.#enabled) {
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

        if (this.#started) {
            for (const e of elementsStarted) {
                e.classList.remove('is-hidden');
            }
            for (const e of elementsStopped) {
                e.classList.add('is-hidden');
            }
        } else {
            for (const e of elementsStarted) {
                e.classList.add('is-hidden');
            }
            for (const e of elementsStopped) {
                e.classList.remove('is-hidden');
            }
        }

        document.getElementById('calibration-switch').checked = this.#enabled;
    }

    async #start() {
        this.#calibCtlSocket.sendMessage(JSON.stringify(['START_CALIBRATION']));
        await this.#waitForReply('STARTED_CALIBRATION');
    }

    async #stop() {
        this.#calibCtlSocket.sendMessage(JSON.stringify(['STOP_CALIBRATION']));
        await this.#waitForReply('STOPPED_CALIBRATION');
    }

    async #reset() {
        this.#calibCtlSocket.sendMessage(JSON.stringify(['RESET_CALIBRATION']));
        await this.#waitForReply('RESET_CALIBRATION');
    }

    async #save() {
        this.#calibCtlSocket.sendMessage(JSON.stringify(['SAVE_CALIBRATION']));
        await this.#waitForReply('SAVED_CALIBRATION');
    }

    async #waitForReply(commandName) {
        while (true) {
            const message = await this.#calibCtlSocket.getNextMessageJson();
            if (message.shift() === commandName) {
                return;
            }
        }
    }
}