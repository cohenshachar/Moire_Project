from flask import Flask, request, jsonify
import numpy as np
import svgwrite
import functions as fun
from svgpathtools import svg2paths
from io import StringIO
from flask_cors import CORS
import ezdxf
import time

app = Flask(__name__)
CORS(app)

def add_ring_to_model(model, type, ring_data, center,constructive = True, paths ='' , old_add = False) -> list:
    if constructive:
        fun.add_grid(model, type, ring_data['inner_radius'], ring_data['outer_radius'], ring_data['repeats'],
                     ring_data['offset'], ring_data['angle'], ring_data['width_s'], ring_data['width_f'],
                     center, ring_data['id'], old_add=old_add)
    else:
        fun.cut_grid(model, type, paths, ring_data['inner_radius'], ring_data['outer_radius'], ring_data['repeats'],
                     ring_data['offset'], ring_data['angle'], ring_data['width_s'], ring_data['width_f'],
                     center)

def create_ring_data(inner_radius, outer_radius, repeats, offset, angle, width_s, width_f,id=''):
    return {'inner_radius': inner_radius, 'outer_radius': outer_radius, 'repeats': repeats, 'offset': offset,
            'angle': angle, 'width_s': width_s,'width_f': width_f, 'id': id}

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        print(data)
        svg_content = data['svg_content']
        svg_viewbox = list(map(float, data['svg_viewbox'].split()))
        rings_radii = data['rings_radii']
        rings = int(data['rings'])
        speed = data['speed']
        rev_width = data['rev_width']
        base_width = data['base_width']
        base_alphas = data['base_alphas']
        info = data['info']
        info_data = data['info_data']
        result = ["", ""]

        inner_radius = rings_radii.pop()
        inner_hour_bound = rings_radii.pop()
        outer_hour_bound = rings_radii.pop()

        size = (f"{rings_radii[0] * 2}px", f"{rings_radii[0] * 2}px")
        center = (rings_radii[0] , rings_radii[0])
        hour_rings = rings - 2
        sum = hour_rings ** 2
        rings_radii.append(outer_hour_bound - (1 + 2 *(hour_rings-1)) * (outer_hour_bound - inner_hour_bound) / sum)
        for i in range(hour_rings-2,0,-1):
            rings_radii.append(rings_radii[-1] - (1 + 2 * i) * (outer_hour_bound - inner_hour_bound) / sum)
        if hour_rings == 1:
            rings_radii.pop()
        rings_radii.append(inner_radius)
        hour_ring_moire_reps = fun.first_n_primes(hour_rings)

        def rings_switch_case(ring):
            h_ring_index = ring - 2
            in_r = rings_radii[ring+1]
            out_r = rings_radii[ring]
            offset = np.pi / 2
            alpha_b = base_alphas[ring]
            nb = speed
            nr = nb-1
            alpha_r = np.degrees(np.arctan((nb / nr) * np.tan(np.radians(alpha_b))))
            ring_base_width = base_width[ring]
            ring_rev_width = rev_width[ring]
            id = ''

            if ring == 0:
                id = "outer"

            elif ring == 1:
                id = "minute"
                nb *= 12
                nr = nb - 1
                alpha_r = np.degrees(np.arctan((nb / nr) * np.tan(np.radians(alpha_b))))

            elif ring >= 2 and ring <= 7:
                id = f"hour_{ring-1}"
                nb *= hour_ring_moire_reps[h_ring_index]
                nr = (speed-1) * hour_ring_moire_reps[h_ring_index]
                alpha_r = np.degrees(np.arctan((nb / nr) * np.tan(np.radians(alpha_b))))
                offset = -(2 * np.pi / (2 * hour_ring_moire_reps[h_ring_index])) - np.pi / 2

            else:
                return "", ""

            base_ring = create_ring_data(in_r, out_r, nb, offset, alpha_b, ring_base_width, ring_base_width, id)
            rev_ring = create_ring_data(in_r, out_r, nr, offset, alpha_r, ring_rev_width, ring_rev_width, id)
            return base_ring, rev_ring

        print(info)
        if info == "full":
            type = 'dxf'
            base_dxf_doc = ezdxf.new('R2000')
            rev_dxf_doc = ezdxf.new('R2000')
            model_b = base_dxf_doc.modelspace()
            model_r = rev_dxf_doc.modelspace()
            svg_file = StringIO(svg_content)
            paths, attributes = svg2paths(svg_file)
            min_x, min_y, width, height = svg_viewbox
            img_center = (width / 2, height / 2)
            fun.check_and_correct(paths, center[0] - img_center[0], center[1] - img_center[1])
            rings_range = range(rings)

            model_b_svg = svgwrite.Drawing('base.svg', size=size)
            model_r_svg = svgwrite.Drawing('revealer.svg', size=size)
            elapsed_time = []
            for i in rings_range:
                ring_data = rings_switch_case(i)
                # add_ring_to_model(model_b, type, ring_data[0],center = center, constructive=False, paths=paths, old_add=True)
                # add_ring_to_model(model_r, type, ring_data[1],center = center, constructive=True, old_add=True)
                start_time = time.time()
                add_ring_to_model(model_b_svg, 'svg', ring_data[0], center=center, constructive=False, paths=paths, old_add=True)
                end_time = time.time()
                elapsed_time.append(end_time - start_time)
                add_ring_to_model(model_r_svg, 'svg', ring_data[1], center=center, constructive=True, old_add=True)
            print("Elapsed Time: ", elapsed_time, ", Total: ", np.sum(elapsed_time))
            model_b_svg.save()
            model_r_svg.save()
            # base_dxf_doc.saveas("base.dxf")
            # rev_dxf_doc.saveas("reveal.dxf")
            # fun.add_crosshair_to_center(base_dxf_model, center)
            # fun.add_crosshair_to_center(revealer_dxf_model, center)
            # base_dxf_doc.saveas("base_center_marked.dxf")
            # rev_dxf_doc.saveas("reveal_center_marked.dxf")
        else:
            type = 'svg'
            model_b = svgwrite.Drawing('base.svg', size=size)
            model_r = svgwrite.Drawing('revealer.svg', size=size)
            rings_range = [info_data]
            if info == "grid":
                rings_range = range(rings)
            elif info == "hour":
                 rings_range = range(2, rings)
            for i in rings_range:
                ring_data = rings_switch_case(i)
                add_ring_to_model(model_b, type, ring_data[0], center=center, constructive=True, old_add=False)
                add_ring_to_model(model_r, type, ring_data[1], center=center, constructive=True, old_add=False)
            result = [model_b.tostring(), model_r.tostring()]

        # Return the result as JSON
        return jsonify({"base": result[0], "revealer": result[1]})
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
