import Viewport from "./Viewport";
import Socket from "./Socket";

export default class Viewports {
    static EVENT_NAME_STREAM_IMAGE = 'socket.stream.received.mat.image';
    static EVENT_NAME_STREAM_RAW = 'socket.stream.received.mat.raw';

    #viewportsElement;
    #streamSocket;
    #viewports = [];


    constructor(viewportsElement) {
        this.#viewportsElement = viewportsElement;
        this.#streamSocket = new Socket('/stream', this.onMessage);

        for (const element in viewportsElement.getElementsByClassName("viewport")) {
            this.#viewports.push(new Viewport(element, this.#streamSocket));
        }
    }

    onMessage(data) {
        let imageData = this.parseData(data);

        if (imageData.codec === 1) {
            const blob = new Blob([frame.pixelData], {type: 'image/jpeg'});
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

    parseData(data) {
        const buffer = data.arrayBuffer();
        const view = new DataView(buffer);

        let offset = 0;
        const nameSize = view.getUint16(offset); offset += 2;
        let name = '';
        for (i = 0; i < nameSize; i++) {
            name += String.fromCharCode(view.getInt8(offset + i))
        }
        offset += nameSize;
        const type = view.getUint8(offset); offset += 1;        // CvMatTypeEnum
        const codec = view.getUint8(offset); offset += 1;       // CvMatCodecEnum
        const channels = view.getUint8(offset); offset += 1;    // число каналов
        const x = view.getUint16(offset, true); offset += 2;
        const y = view.getUint16(offset, true); offset += 2;
        const w = view.getUint16(offset, true); offset += 2;
        const h = view.getUint16(offset, true); offset += 2;
        const scale = view.getFloat32(offset, true); offset += 4;

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

        // Проверяем размер данных изображения
        const expectedDataSize = w * h * channels * bytesPerChannel;
        const availableSize = buffer.byteLength - offset;
        if (availableSize < expectedDataSize) {
            throw new Error(
                `Incomplete image data: expected ${expectedDataSize} bytes, ` +
                `got ${availableSize}`
            );
        }

        // Извлекаем сырые пиксели (ArrayBuffer)
        const pixelData = buffer.slice(offset, offset + expectedDataSize);

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
                scale,
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
    /**
     * Преобразует объект с данными в бинарный ArrayBuffer в соответствии со схемой типов.
     * @param {Object} data - объект, где ключи соответствуют полям схемы, значения - числа
     * @param {Object} schema - объект, где ключи - имена полей, значения - строки типа ('int', 'uint', 'short', 'ushort', 'char', 'uchar', 'float', 'double')
     * @returns {ArrayBuffer} - бинарное представление данных (little-endian)
     */
    toBinaryStruct(data, schema) {
        // Размеры типов в байтах
        const typeSizes = {
            'int': 4,
            'uint': 4,
            'short': 2,
            'ushort': 2,
            'char': 1,
            'uchar': 1,
            'float': 4,
            'double': 8
        };

        // Вычисляем общий размер и собираем информацию о полях в порядке схемы
        let totalSize = 0;
        const fields = [];
        for (const key in schema) {
            if (schema.hasOwnProperty(key)) {
                const type = schema[key];
                const size = typeSizes[type];
                if (size === undefined) {
                    throw new Error(`Неподдерживаемый тип: ${type}`);
                }
                fields.push({ key, type, offset: totalSize });
                totalSize += size;
            }
        }

        // Создаём буфер и DataView
        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);

        // Записываем каждое поле
        for (const field of fields) {
            const value = data[field.key];
            if (typeof value !== 'number') {
                throw new Error(`Поле ${field.key} должно быть числом`);
            }

            switch (field.type) {
                case 'int':
                    view.setInt32(field.offset, value, true);
                    break;
                case 'uint':
                    view.setUint32(field.offset, value, true);
                    break;
                case 'short':
                    view.setInt16(field.offset, value, true);
                    break;
                case 'ushort':
                    view.setUint16(field.offset, value, true);
                    break;
                case 'char':
                    view.setInt8(field.offset, value);
                    break;
                case 'uchar':
                    view.setUint8(field.offset, value);
                    break;
                case 'float':
                    view.setFloat32(field.offset, value, true);
                    break;
                case 'double':
                    view.setFloat64(field.offset, value, true);
                    break;
                default:
                    throw new Error(`Неподдерживаемый тип: ${field.type}`);
            }
        }

        return buffer;
    }
}