import Viewports from "./Viewports.js";
import Telemetry from "./Telemetry.js";
import Logger from "./Logger.js";
import Status from "./Status.js";
import CalibrationMode from "./CalibrationMode.js";
import Socket from "./Socket.js";

document.onreadystatechange = function() {
    window.logger = new Logger(document.getElementById("logs-container"));
    window.telemetry = new Telemetry();
    window.telemetryStatus = new Status();
    window.viewports = new Viewports(document.getElementById("app-viewports"));
    const cvCtl = new Socket("/cv_ctl", (data) => {
        document.addEventListener('socket.cvCtl.onmessage', new CustomEvent({detail: {data}}));
    });

    cvCtl.connect();

    window.calibrationMode = new CalibrationMode(cvCtl);
}