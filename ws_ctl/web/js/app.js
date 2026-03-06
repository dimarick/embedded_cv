import Viewports from "./Viewports.js";
import Telemetry from "./Telemetry.js";
import Logger from "./Logger.js";
import Status from "./Status.js";

document.onreadystatechange = function() {
    window.logger = new Logger(document.getElementById("logs-container"));
    window.telemetry = new Telemetry();
    window.telemetryStatus = new Status();
    window.viewports = new Viewports(document.getElementById("app-viewports"));
}