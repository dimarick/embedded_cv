import Viewport from "./Viewport.js";
import Socket from "./Socket.js";

export default class Viewports {
    static EVENT_NAME_STREAM_IMAGE = 'socket.stream.received.mat.image';
    static EVENT_NAME_STREAM_RAW = 'socket.stream.received.mat.raw';

    #viewportsElement;
    #streamSocket;
    #viewports = [];


    constructor(viewportsElement) {
        this.#viewportsElement = viewportsElement;
        this.#streamSocket = new Socket('/stream', (data) => this.onMessage(data));

        for (const element of viewportsElement.getElementsByClassName("viewport")) {
            this.#viewports.push(new Viewport(element, this.#streamSocket));
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
    fourCC(str) {
        if (str.length !== 4) throw new Error('FourCC must be 4 chars');
        return (str.charCodeAt(0) << 24) |
            (str.charCodeAt(1) << 16) |
            (str.charCodeAt(2) << 8)  |
            str.charCodeAt(3);
    }
    // /**
    //  * Преобразует объект с данными в бинарный ArrayBuffer в соответствии со схемой типов.
    //  * @param {Object} data - объект, где ключи соответствуют полям схемы, значения - числа
    //  * @param {Object} schema - объект, где ключи - имена полей, значения - строки типа ('int', 'uint', 'short', 'ushort', 'char', 'uchar', 'float', 'double')
    //  * @returns {ArrayBuffer} - бинарное представление данных (little-endian)
    //  */
    // toBinaryStruct(data, schema) {
    //     // Размеры типов в байтах
    //     const typeSizes = {
    //         'int': 4,
    //         'uint': 4,
    //         'short': 2,
    //         'ushort': 2,
    //         'char': 1,
    //         'uchar': 1,
    //         'float': 4,
    //         'double': 8
    //     };
    //
    //     // Вычисляем общий размер и собираем информацию о полях в порядке схемы
    //     let totalSize = 0;
    //     const fields = [];
    //     for (const key of schema) {
    //         if (schema.hasOwnProperty(key)) {
    //             const type = schema[key];
    //             const size = typeSizes[type];
    //             if (size === undefined) {
    //                 throw new Error(`Неподдерживаемый тип: ${type}`);
    //             }
    //             fields.push({ key, type, offset: totalSize });
    //             totalSize += size;
    //         }
    //     }
    //
    //     // Создаём буфер и DataView
    //     const buffer = new ArrayBuffer(totalSize);
    //     const view = new DataView(buffer);
    //
    //     // Записываем каждое поле
    //     for (const field of fields) {
    //         const value = data[field.key];
    //         if (typeof value !== 'number') {
    //             throw new Error(`Поле ${field.key} должно быть числом`);
    //         }
    //
    //         switch (field.type) {
    //             case 'int':
    //                 view.setInt32(field.offset, value, true);
    //                 break;
    //             case 'uint':
    //                 view.setUint32(field.offset, value, true);
    //                 break;
    //             case 'short':
    //                 view.setInt16(field.offset, value, true);
    //                 break;
    //             case 'ushort':
    //                 view.setUint16(field.offset, value, true);
    //                 break;
    //             case 'char':
    //                 view.setInt8(field.offset, value);
    //                 break;
    //             case 'uchar':
    //                 view.setUint8(field.offset, value);
    //                 break;
    //             case 'float':
    //                 view.setFloat32(field.offset, value, true);
    //                 break;
    //             case 'double':
    //                 view.setFloat64(field.offset, value, true);
    //                 break;
    //             default:
    //                 throw new Error(`Неподдерживаемый тип: ${field.type}`);
    //         }
    //     }
    //
    //     return buffer;
    // }
}