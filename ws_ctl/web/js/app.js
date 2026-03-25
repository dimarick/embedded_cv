// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

import Viewports from "./Viewports.js";
import Telemetry from "./Telemetry.js";
import Logger from "./Logger.js";
import Status from "./Status.js";
import CalibrationMode from "./CalibrationMode.js";
import Socket from "./Socket.js";

document.onreadystatechange = function() {
    if (document.readyState !== 'complete') {
        return;
    }
    const cvCtl = new Socket("/cv_ctl", async (data) => {
        if (data instanceof Blob) {
            const d = await data.text();
            document.dispatchEvent(new CustomEvent('socket.cvCtl.onmessage', {detail: d}));
            return
        }
        document.dispatchEvent(new CustomEvent('socket.cvCtl.onmessage', {detail: data}));
    });

    const calibCtl = new Socket("/cv_calib_ctl", async (data) => {
        if (data instanceof Blob) {
            const d = await data.text();
            document.dispatchEvent(new CustomEvent('socket.calibCtl.onmessage', {detail: d}));
            return
        }
        document.dispatchEvent(new CustomEvent('socket.cvCtl.onmessage', {detail: data}));
    });

    cvCtl.connect();
    calibCtl.connect();

    window.logger = new Logger(document.getElementById("logs-container"));
    window.telemetry = new Telemetry();
    window.telemetryStatus = new Status();
    window.viewports = new Viewports(document.getElementById("app-viewports"));

    window.calibrationMode = new CalibrationMode(cvCtl, calibCtl);
}