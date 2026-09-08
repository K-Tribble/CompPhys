#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "types.hpp"

namespace mcmc {

namespace io {

    // Writes to .npy files to easily load with numpy

    template <typename T>
    inline constexpr const char* npyDescr() {
        if constexpr (std::is_same_v<T, double>) {
            return "<f8";
        } else if constexpr (std::is_same_v<T, float>) {
            return "<f4";
        } else if constexpr (std::is_same_v<T, std::int8_t>) {
            return "|i1";
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
            return "<i4";
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
            return "<i8";
        } else {
            static_assert(sizeof(T) == 0, "npyDescr: unsupported element type");
        }
    }

    template <typename T>
    inline void writeNpy(const std::string& path, std::span<const T> data, std::span<const u32> shape) {
        u32 expected = 1;
        for (const u32 d : shape) {
            expected *= d;
        }

        if (expected != data.size()) {
            throw std::invalid_argument("writeNpy: shape does not match data size");
        }

        std::ostringstream dict;
        dict << "{'descr': '" << npyDescr<T>()
            << "', 'fortran_order': False, 'shape': (";
        for (const u32 d : shape) {
            dict << d << ", ";
        }
        dict << "), }";
 
        std::string header = dict.str();

        // magic(6) + version(2) + headerLen(2) + header + '\n' must be a multiple
        // of 64 bytes, so the raw data starts aligned
        const u32 prefix = 6 + 2 + 2;
        u32 total = prefix + header.size() + 1;
        const u32 pad = (64 - (total % 64)) % 64;
        header.append(pad, ' ');
        header.push_back('\n');

        const std::uint16_t headerLen = static_cast<std::uint16_t>(header.size());

        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("writeNpy: cannot open " + path);
        }
    
        out.write("\x93NUMPY", 6);
        const char version[2] = {'\x01', '\x00'};
        out.write(version, 2);
        out.write(reinterpret_cast<const char*>(&headerLen), 2);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(T)));
 
        if (!out) {
            throw std::runtime_error("writeNpy: write failed for " + path);
        }
    }

    // Minimal JSON sidecar
    // One of these per run, next to .npy
    // A temperature scan then becomes a
    // directory of file pairs, and the whole scan loads in Python with
    // pd.DataFrame([json.load(open(f)) for f in sorted(glob("*.json"))])

    class JsonWriter {
    public:
        JsonWriter() { os_ << "{\n"; }
 
        JsonWriter& add(const std::string& key, d64 value) {
            comma();
            os_ << "  " << quote(key) << ": ";
            if (std::isnan(value)) {
                os_ << "null";
            } else if (std::isinf(value)) {
                os_ << (value > 0 ? "1e999" : "-1e999");
            } else {
                os_ << std::setprecision(17) << value;
            }
            return *this;
        }
 
        JsonWriter& add(const std::string& key, u32 value) {
            comma();
            os_ << "  " << quote(key) << ": " << value;
            return *this;
        }
    
        JsonWriter& add(const std::string& key, bool value) {
            comma();
            os_ << "  " << quote(key) << ": " << (value ? "true" : "false");
            return *this;
        }
 
        JsonWriter& add(const std::string& key, const std::string& value) {
            comma();
            os_ << "  " << quote(key) << ": " << quote(value);
            return *this;
        }
    
        JsonWriter& add(const std::string& key, const std::vector<d64>& v) {
            comma();
            os_ << "  " << quote(key) << ": [";
            for (u32 i = 0; i < v.size(); ++i) {
                os_ << (i ? ", " : "");
                if (std::isnan(v[i])) {
                    os_ << "null";
                } else {
                    os_ << std::setprecision(17) << v[i];
                }
            }
            os_ << "]";
            return *this;
        }

        JsonWriter& add(const std::string& key, const std::vector<std::string>& v) {
            comma();
            os_ << "  " << quote(key) << ": [";
            for (u32 i = 0; i < v.size(); ++i) {
                os_ << (i ? ", " : "") << quote(v[i]);
            }
            os_ << "]";
            return *this;
        }
 
        std::string str() const { return os_.str() + "\n}\n"; }
 
        void write(const std::string& path) const {
            std::ofstream out(path);
            if (!out) {
                throw std::runtime_error("JsonWriter: cannot open " + path);
            }
            out << str();
    }
 
    private:
        void comma() {
            if (!first_) {
                os_ << ",\n";
            }
            first_ = false;
        }
 
        static std::string quote(const std::string& s) { return "\"" + s + "\""; }
 
        std::ostringstream os_;
        bool first_ = true;
};


} // namespace io

} // namespace mcmc
