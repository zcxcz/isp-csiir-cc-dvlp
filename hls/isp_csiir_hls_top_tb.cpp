//==============================================================================
// ISP-CSIIR HLS Top Module Testbench
//==============================================================================

#ifndef ISP_CSIIR_HLS_TOP_TB_HPP
#define ISP_CSIIR_HLS_TOP_TB_HPP

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "isp_csiir_hls_top.cpp"

using namespace std;

namespace {

string read_text_file(const string& filename) {
    ifstream handle(filename.c_str());
    if (!handle.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    return string((istreambuf_iterator<char>(handle)), istreambuf_iterator<char>());
}

int parse_json_int(const string& content, const string& key, int default_value) {
    const string token = "\"" + key + "\"";
    size_t pos = content.find(token);
    if (pos == string::npos) {
        return default_value;
    }
    pos = content.find(':', pos);
    if (pos == string::npos) {
        return default_value;
    }
    pos++;
    while (pos < content.size() && isspace(static_cast<unsigned char>(content[pos]))) {
        pos++;
    }
    char* end_ptr = nullptr;
    long value = strtol(content.c_str() + pos, &end_ptr, 10);
    if (end_ptr == content.c_str() + pos) {
        return default_value;
    }
    return static_cast<int>(value);
}

string parse_json_string(const string& content, const string& key, const string& default_value) {
    const string token = "\"" + key + "\"";
    size_t pos = content.find(token);
    if (pos == string::npos) {
        return default_value;
    }
    pos = content.find(':', pos);
    if (pos == string::npos) {
        return default_value;
    }
    pos = content.find('"', pos + 1);
    if (pos == string::npos) {
        return default_value;
    }
    size_t end = content.find('"', pos + 1);
    if (end == string::npos) {
        return default_value;
    }
    return content.substr(pos + 1, end - pos - 1);
}

void ensure_size_fits(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw runtime_error("Image width/height must be positive");
    }
    if (width > MAX_WIDTH || height > MAX_HEIGHT) {
        ostringstream oss;
        oss << "Image size " << width << "x" << height
            << " exceeds HLS top limits " << MAX_WIDTH << "x" << MAX_HEIGHT;
        throw runtime_error(oss.str());
    }
}

vector<pixel_t> load_input(const string& filename, int width, int height) {
    ensure_size_fits(width, height);
    ifstream f(filename.c_str());
    if (!f.is_open()) {
        throw runtime_error("Cannot open input file: " + filename);
    }

    vector<pixel_t> img(static_cast<size_t>(width) * static_cast<size_t>(height));
    string line;
    int idx = 0;
    while (getline(f, line) && idx < width * height) {
        if (line.empty() || line[0] == '#' || line[0] == '/') {
            continue;
        }
        istringstream iss(line);
        int val = 0;
        if (iss >> hex >> val) {
            img[static_cast<size_t>(idx++)] = val;
        }
    }
    if (idx != width * height) {
        ostringstream oss;
        oss << "Input sample count mismatch in " << filename
            << ": got " << idx << ", expected " << (width * height);
        throw runtime_error(oss.str());
    }
    return img;
}

void save_output(const string& filename, const vector<pixel_t>& output) {
    filesystem::path path(filename);
    if (path.has_parent_path()) {
        filesystem::create_directories(path.parent_path());
    }
    ofstream f(filename.c_str());
    if (!f.is_open()) {
        throw runtime_error("Cannot open output file: " + filename);
    }
    for (size_t i = 0; i < output.size(); i++) {
        f << hex << output[i].to_int() << "\n";
    }
}

void load_reg_cfg(const string& filename, ISPCSIIR_Regs& regs) {
    const string content = read_text_file(filename);
    regs.img_width = parse_json_int(content, "reg_image_width", regs.img_width.to_int());
    regs.img_height = parse_json_int(content, "reg_image_height", regs.img_height.to_int());
    regs.win_size_thresh[0] = parse_json_int(content, "reg_win_size_thresh_0", regs.win_size_thresh[0].to_int());
    regs.win_size_thresh[1] = parse_json_int(content, "reg_win_size_thresh_1", regs.win_size_thresh[1].to_int());
    regs.win_size_thresh[2] = parse_json_int(content, "reg_win_size_thresh_2", regs.win_size_thresh[2].to_int());
    regs.win_size_thresh[3] = parse_json_int(content, "reg_win_size_thresh_3", regs.win_size_thresh[3].to_int());
    regs.win_size_clip_y[0] = parse_json_int(content, "reg_win_size_clip_y_0", regs.win_size_clip_y[0].to_int());
    regs.win_size_clip_y[1] = parse_json_int(content, "reg_win_size_clip_y_1", regs.win_size_clip_y[1].to_int());
    regs.win_size_clip_y[2] = parse_json_int(content, "reg_win_size_clip_y_2", regs.win_size_clip_y[2].to_int());
    regs.win_size_clip_y[3] = parse_json_int(content, "reg_win_size_clip_y_3", regs.win_size_clip_y[3].to_int());
    regs.win_size_clip_sft[0] = parse_json_int(content, "reg_win_size_clip_sft_0", regs.win_size_clip_sft[0].to_int());
    regs.win_size_clip_sft[1] = parse_json_int(content, "reg_win_size_clip_sft_1", regs.win_size_clip_sft[1].to_int());
    regs.win_size_clip_sft[2] = parse_json_int(content, "reg_win_size_clip_sft_2", regs.win_size_clip_sft[2].to_int());
    regs.win_size_clip_sft[3] = parse_json_int(content, "reg_win_size_clip_sft_3", regs.win_size_clip_sft[3].to_int());
    regs.blending_ratio[0] = parse_json_int(content, "reg_blending_ratio_0", regs.blending_ratio[0].to_int());
    regs.blending_ratio[1] = parse_json_int(content, "reg_blending_ratio_1", regs.blending_ratio[1].to_int());
    regs.blending_ratio[2] = parse_json_int(content, "reg_blending_ratio_2", regs.blending_ratio[2].to_int());
    regs.blending_ratio[3] = parse_json_int(content, "reg_blending_ratio_3", regs.blending_ratio[3].to_int());
    regs.edge_protect = parse_json_int(content, "reg_edge_protect", regs.edge_protect.to_int());
}

struct ImgInfo {
    int width;
    int height;
    string input_path;
    string hls_output_path;
};

ImgInfo load_img_info(const string& filename) {
    const string content = read_text_file(filename);
    ImgInfo info;
    info.width = parse_json_int(content, "width", 64);
    info.height = parse_json_int(content, "height", 64);
    info.input_path = parse_json_string(content, "input_path", "");
    info.hls_output_path = parse_json_string(content, "hls_pattern_path", "");
    if (info.input_path.empty()) {
        throw runtime_error("img_info.json missing image.input_path");
    }
    if (info.hls_output_path.empty()) {
        throw runtime_error("img_info.json missing outputs.hls_pattern_path");
    }
    return info;
}

vector<pixel_t> process_top(const vector<pixel_t>& input, int width, int height, ISPCSIIR_Regs& regs) {
    csiir_hls::stream_t<axis_pixel_t> din_stream;
    csiir_hls::stream_t<axis_pixel_t> dout_stream;
    vector<pixel_t> output(static_cast<size_t>(width) * static_cast<size_t>(height));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            axis_pixel_t din;
            din.data = input[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
            din.last = (y == height - 1 && x == width - 1) ? 1 : 0;
            din.user = (y == 0 && x == 0) ? 1 : 0;
            din_stream.write(din);
        }
    }

    isp_csiir_top(din_stream, dout_stream, regs);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            axis_pixel_t dout = dout_stream.read();
            output[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = dout.data;
        }
    }
    return output;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        string reg_cfg_file = (argc >= 2) ? argv[1] : "config/reg_cfg_seed0.json";
        string img_info_file = (argc >= 3) ? argv[2] : "config/img_info.json";
        string output_override = (argc >= 4) ? argv[3] : "";

        ISPCSIIR_Regs regs;
        regs.reset();
        load_reg_cfg(reg_cfg_file, regs);

        ImgInfo img_info = load_img_info(img_info_file);
        if (img_info.width != regs.img_width.to_int() || img_info.height != regs.img_height.to_int()) {
            throw runtime_error("reg_cfg image size does not match img_info image size");
        }

        const string output_path = output_override.empty() ? img_info.hls_output_path : output_override;
        vector<pixel_t> input = load_input(img_info.input_path, img_info.width, img_info.height);
        vector<pixel_t> output = process_top(input, img_info.width, img_info.height, regs);
        save_output(output_path, output);

        int min_in = 1024;
        int max_in = 0;
        for (size_t i = 0; i < input.size(); i++) {
            int val = input[i].to_int();
            min_in = min(min_in, val);
            max_in = max(max_in, val);
        }

        int min_out = 1024;
        int max_out = 0;
        for (size_t i = 0; i < output.size(); i++) {
            int val = output[i].to_int();
            min_out = min(min_out, val);
            max_out = max(max_out, val);
        }

        cout << "ISP-CSIIR HLS Top Test" << endl;
        cout << "======================" << endl;
        cout << "Register config: " << reg_cfg_file << endl;
        cout << "Image info:      " << img_info_file << endl;
        cout << "Image size:      " << img_info.width << " x " << img_info.height << endl;
        cout << "Input range:     [" << min_in << ", " << max_in << "]" << endl;
        cout << "Output range:    [" << min_out << ", " << max_out << "]" << endl;
        cout << "Output written:  " << output_path << endl;
        return 0;
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }
}

#endif  // ISP_CSIIR_HLS_TOP_TB_HPP
