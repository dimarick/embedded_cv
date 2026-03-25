// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

import Viewport from "./Viewport.js";
import Socket from "./Socket.js";

export default class Viewports {
    static EVENT_NAME_STREAM_IMAGE = 'stream.received.mat.image';
    static EVENT_NAME_STREAM_RAW = 'stream.received.mat.raw';

    #viewportsElement;
    #streamSocket;
    #streamCalibSocket;
    #viewports = [];

    constructor(viewportsElement) {
        this.#viewportsElement = viewportsElement;
        this.#streamSocket = new Socket('/cv_stream', (data) => this.onMessage(data));
        this.#streamCalibSocket = new Socket('/cv_calib_stream', (data) => this.onMessage(data));
        this.#streamSocket.connect();
        this.#streamCalibSocket.connect();

        for (const element of viewportsElement.getElementsByClassName("viewport")) {
            this.#viewports.push(new Viewport(element, {cv: this.#streamSocket, calib: this.#streamCalibSocket}));
        }
    }

    async onMessage(data) {
        let imageData = await this.parseData(data);

        if (imageData.header.codec === 1) {
            const blob = new Blob([imageData.pixelData], {type: 'image/jpeg'});
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                imageData.image = img;
                document.dispatchEvent(new CustomEvent(Viewports.EVENT_NAME_STREAM_IMAGE, {detail: imageData}));
            };
            img.onerror = (event) => {
                console.log(event);
            };
            img.src = url;
        } else {
            document.dispatchEvent(new CustomEvent(Viewports.EVENT_NAME_STREAM_RAW, {detail: imageData}));
        }
    }

    async parseData(data) {
        const buffer = await data.arrayBuffer();
        const view = new DataView(buffer);

        let offset = 0;
        const nameSize = view.getUint16(offset, true); offset += 2;
        let name = '';
        for (let i = 0; i < nameSize; i++) {
            name += String.fromCharCode(view.getInt8(offset + i))
        }
        offset += nameSize;
        const type = view.getInt8(offset); offset += 1;        // CvMatTypeEnum
        const codec = view.getInt8(offset); offset += 1;       // CvMatCodecEnum
        const channels = view.getInt16(offset, true); offset += 2;    // число каналов
        const viewW = view.getInt16(offset, true); offset += 2;
        const viewH = view.getInt16(offset, true); offset += 2;
        const x = view.getInt16(offset, true); offset += 2;
        const y = view.getInt16(offset, true); offset += 2;
        const w = view.getInt16(offset, true); offset += 2;
        const h = view.getInt16(offset, true); offset += 2;

        // Определяем количество байт на один канал по типу данных
        const typeToBytesPerChannel = {
            0x10: 1, // TYPE_8U
            0x11: 1, // TYPE_8S
            0x20: 2, // TYPE_16U
            0x21: 2, // TYPE_16S
            0x22: 2, // TYPE_16F
            0x32: 4, // TYPE_32F
            0x42: 8, // TYPE_64F
        };
        const bytesPerChannel = typeToBytesPerChannel[type];
        if (bytesPerChannel === undefined) {
            throw new Error(`Unsupported image type: 0x${type.toString(16)}`);
        }

        // Извлекаем сырые пиксели (ArrayBuffer)
        const pixelData = buffer.slice(offset);

        // Возвращаем структурированный результат
        return {
            header: {
                name,
                type,
                codec,
                channels,
                x,
                y,
                w,
                h,
                viewW,
                viewH,
            },
            pixelData,
        };
    }
}