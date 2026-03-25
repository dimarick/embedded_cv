// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

import ViewportStream from "./ViewportStream.js";
import Viewports from "./Viewports.js";

export default class ViewportInteractiveStream extends ViewportStream {
    #infoImageName;
    #infoImage;
    #valueElement;
    #fragmentSize = {w: 128, h: 128};

    constructor(canvas, socket, valueElement) {
        super(canvas, socket);
        this.#infoImageName = canvas.dataset.infoImage;
        this.#valueElement = valueElement;

        canvas.addEventListener('mousemove', (event) => {
            const x = event.offsetX;
            const y = event.offsetY;

            this.#loadHoverFragment(x, y);
            this.#showHoverValue(x, y);
        })

        this.element.parentNode.addEventListener('click', (event) => {

            if (event.target.classList.contains('interactive-viewport-value-tmp')) {
                this.element.parentNode.removeChild(event.target);
                return;
            }

            const x = event.offsetX;
            const y = event.offsetY;
            this.#addValue(x, y);
        })

        canvas.addEventListener('mouseout', (event) => {
            this.#unloadHoverFragment();
        })

        document.addEventListener(Viewports.EVENT_NAME_STREAM_RAW, (event) => this.onRawImage(event))
    }

    onRawImage(event) {
        if (event.detail.header.name === this.#infoImageName) {
            const header = event.detail.header;
            if (header.codec === 0 && header.type === 0x32 && header.channels === 1) {
                this.#infoImage = event.detail;
            }
        }
    }

    #loadHoverFragment(x, y) {
        if (!this.imageHeader) {
            return;
        }
        const rect = this.element.getBoundingClientRect();
        const imageX = x / rect.width * this.imageHeader.w - this.#fragmentSize.w / 2;
        const imageY = y / rect.height * this.imageHeader.h - this.#fragmentSize.h / 2;

        this.socket.sendMessage("CHANNEL " + this.quote(this.#infoImageName) +
            " 0 " + this.#fragmentSize.w + " " + this.#fragmentSize.h +
            " " + Math.round(Math.min(this.imageHeader.w - this.#fragmentSize.w, Math.max(0, imageX))) +
            " " + Math.round(Math.min(this.imageHeader.h - this.#fragmentSize.h, Math.max(0, imageY))) + " " +
            this.#fragmentSize.w + " " + this.#fragmentSize.h
        );
    }

    #showHoverValue(x, y) {
        if (!this.#infoImage) {
            return;
        }
        const value = this.#getValue(x, y);
        this.#valueElement.innerHTML = this.#formatValue(value);
    }

    #formatValue(value) {
        return !!value ? Math.round(value * 100) / 100 + ' см' : '';
    }

    #unloadHoverFragment(x, y) {
        this.#valueElement.innerHTML = '';
        this.socket.sendMessage("DESTROY_CHANNEL " + super.quote(this.#infoImageName) + " 0");
    }

    #addValue(x, y) {
        if (!this.#infoImage) {
            return;
        }
        const value = this.#getValue(x, y);

        const element = document.createElement('div');
        element.className = 'interactive-viewport-value interactive-viewport-value-tmp';
        element.style.position = 'absolute';
        element.style.top = y + 'px';   // обычно top — это y, left — x
        element.style.left = x + 'px';
        element.innerHTML = `<i style="transform: translate(-50%, -50%)" class="fa fa-crosshairs" aria-hidden="true"></i>${this.#formatValue(value)}`;
        this.element.parentNode.appendChild(element);
    }

    #getValue(x, y) {
        const rect = this.element.getBoundingClientRect();
        const patchOffsetX = this.#infoImage.header.x;
        const patchOffsetY = this.#infoImage.header.y;
        const view = new DataView(this.#infoImage.pixelData);
        const imageX = x / rect.width * this.imageHeader.w;
        const imageY = y / rect.height * this.imageHeader.h;
        const patchX = imageX - patchOffsetX;
        const patchY = imageY - patchOffsetY;

        // поддерживается только RAW CV_32FC1
        if (patchY < 0 || patchY > this.#infoImage.header.h - 4 || patchX < 0 || patchX > this.#infoImage.header.w - 4) {
            this.#valueElement.innerHTML = '';
            return null;
        }

        let pixels = [];
        for (let i = -2; i <= 2; i++) {
            for (let j = -2; j <= 2; j++) {
                const pixel = view.getFloat32(Math.round(((patchY + j) * this.#infoImage.header.w + patchX + i)) * 4, true)
                if (pixel < 0.01) {
                    continue;
                }
                pixels.push(pixel)
            }
        }

        pixels.sort();

        return pixels[Math.floor(pixels.length / 2)];
    }
}