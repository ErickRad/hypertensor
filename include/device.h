#ifndef HYPERTENSOR_DEVICE_H
#define HYPERTENSOR_DEVICE_H

#include <cstdint>

namespace hypertensor {

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;
    bool operator==(const Device& o) const { return type==o.type && index==o.index; }
    bool operator!=(const Device& o) const { return !(*this==o); }
};

} // namespace hypertensor

#endif // HYPERTENSOR_DEVICE_H