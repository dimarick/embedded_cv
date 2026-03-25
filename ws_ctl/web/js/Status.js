// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

import Telemetry from "./Telemetry.js";
export default class Status {
    #state = {};
    constructor() {
        document.addEventListener(Telemetry.EVENT_NAME_TELEMETRY, (event) => this.onTelemetry(event.detail));
    }

    onTelemetry(event) {
        if (event.command !== 'STATUS') {
            return;
        }

        const component = event.args.shift();
        this.setStatusAll(component, event.args);
    }

    setStatusAll(component, args) {
        while (args.length >= 2) {
            const property = args.shift();
            const value = args.shift();
            this.setStatus(component, property, value);
        }
    }

    setStatus(component, property, value) {
        const tm = component + '.' + property;
        const prev = this.#state[tm];

        if (value === prev) {
            return;
        }

        for (const element of document.querySelectorAll('.telemetry-value[name="' + tm + '"]')) {
            element.innerHTML = value;
        }

        document.dispatchEvent(new CustomEvent(tm, {detail: {component, property, value}}));
        console.log("STATUS", component, property, value);
    }
}