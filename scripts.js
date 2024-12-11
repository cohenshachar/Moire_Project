    let loaded = false;
    let isPlaying = true;
    let direction = 1; // Initial direction

// Function to set the rotation degrees for an element
function setRotationDegrees(degrees) {
    document.documentElement.style.setProperty('--rotation-deg', `${degrees}deg`);
}

// Function to set the rotation duration for an element
function setRotationTime(newDuration) {
    document.documentElement.style.setProperty('--animation-dur', `${newDuration}s`);
}

//------// SITE LISTENERS //-----//

// Event listener for when the DOM content is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    loadEvents();
});

// Function to load demo control events
function loadDemoControlsEvents() {
    const rangeInput = document.getElementById('range');
    const stopButton = document.getElementById('stop');
    const playButton = document.getElementById('play');
    const changeDirectionButton = document.getElementById('change-direction');

    // Event listener for range input to adjust rotation speed
    rangeInput.addEventListener('input', () => {
        const speedMultiplier = rangeInput.value;
        setRotationTime(12 * 60 * 60 / speedMultiplier);  // Set rotation time based on speed multiplier
    });

    // Event listener for stop button to pause the rotation
    stopButton.addEventListener('click', () => {
        const element = document.querySelector('.rotate');
        if (element) {
            element.style.animationPlayState = 'paused';
            isPlaying = false;
        }
    });

    // Event listener for play button to resume the rotation
    playButton.addEventListener('click', () => {
        const element = document.querySelector('.rotate');
        if (element) {
            if (!isPlaying) {
                element.style.animationPlayState = 'running';
                isPlaying = true;
            }
        }
    });

    // Event listener for change direction button to reverse the rotation direction
    changeDirectionButton.addEventListener('click', () => {
        const element = document.querySelector('.rotate');
        if (element) {
            direction *= -1;
            element.style.animationDirection = direction === 1 ? 'normal' : 'reverse';
        }
    });
}
// Function to load event listeners for number input changes
function loadDimensionNumberInputsEvent() {
    const inputs = document.querySelectorAll('.form-container input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('change', (event) => {
            const color = event.target.getAttribute('data-target');
            handleDiameterChange(event.target.id, color);  // Handle the change in diameter
        });
    });
}

// Function to load event listeners for toggle view buttons
function loadDimensionToggleViewEvent() {
    const buttons = document.querySelectorAll('.form-container .toggle-view');
    buttons.forEach(button => {
        button.addEventListener('click', (event) => {
            const input = event.target.previousElementSibling;
            const color = input.getAttribute('data-target');
            toggleView(input.id, color);  // Toggle the view based on the input ID and color
        });
    });
}

// Function to load event listeners for tab buttons
function loadTabsEvent() {
    document.querySelectorAll('.tabs button').forEach(button => {
        if (button.textContent !== "Save") {
            button.addEventListener('click', () => {
                const target = button.getAttribute('data-target');
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');  // Remove active class from all tabs
                });
                const targetElement = document.querySelector(target);
                targetElement.classList.add('active');  // Add active class to the target tab
                const targetId = targetElement.getAttribute('id');
                switch (targetId){
                    case 'tab3':
                        loadDemo();
                        break;
                    default:
                }
            });
        } else {
            button.addEventListener('click', () => {
                sendMoireData('full');  // Send Moire data when Save button is clicked
            });
        }
    });
}
// Function to load event listeners for picture tabs
function loadPictureTabsEvent(){
    document.querySelectorAll('.svg-selectors button').forEach(button => {
        // Clone the button to remove any existing event listeners
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);

        // Add a click event listener to the new button
        newButton.addEventListener('click', () => {
            const revealer = document.getElementById('revealer_svg');
            const target = button.getAttribute('data-target');
            const targetElement = document.querySelector(target);
            const demoLoaded = revealer.classList.contains('rotate');
            revealer.classList.remove('rotate');

            switch (target){
                case '#dimensions_svg':
                    const visible = isSvgVisible(targetElement);
                    if (!visible) {
                        // Show the SVG if it is not visible
                        showSVG(targetElement);
                        newButton.textContent = 'Dimensions (Viewed)';
                    } else {
                        // Hide the SVG if it is visible
                        hideSVG(targetElement);
                        newButton.textContent = 'Dimensions (Hidden)';
                    }
                    if (demoLoaded)
                        revealer.classList.add('rotate');
                    break;
                case '#input_svg':
                case '#revealer_svg':
                case '#base_svg':
                    // Handle visibility for input, revealer, and base SVGs
                    handleSvgVisibility(targetElement);
                    break;
                case '#demo_svg':
                    // Load the demo SVG
                    loadDemo();
                    break;
                default:
                    // Do nothing for unrecognized targets
            }
        });
    });
}

 // Function to load event listeners for ring parameter changes
function loadRingParamEvent() {
    // Event listener for speed change
    document.getElementById('speed').addEventListener('change', () => {
        sendMoireData('grid');  // Send grid data when speed changes
    });

    // Event listeners for minute ring inputs
    document.getElementById('minute_ring').querySelectorAll('input').forEach(input => {
        input.addEventListener('change', () => {
            sendMoireData('ring', 1);  // Send data for minute ring when inputs change
        });
    });

    // Event listeners for outer ring inputs
    document.getElementById('outer_ring').querySelectorAll('input').forEach(input => {
        input.addEventListener('change', () => {
            sendMoireData('ring', 0);  // Send data for outer ring when inputs change
        });
    });

    // Event listeners for mouse leave on ring parameters
    document.querySelectorAll('.ring-param').forEach(ring => {
        ring.addEventListener('mouseleave', function () {
            changePathColor(this.dataset.target, 'black');  // Change path color to black on mouse leave
        });
    });

    // Event listeners for mouse enter on ring parameters
    document.querySelectorAll('.ring-param').forEach(ring => {
        ring.addEventListener('mouseenter', function () {
            changePathColor(this.dataset.target, 'red');  // Change path color to red on mouse enter
        });
    });

    // Event listeners for hour ring radio buttons
    const radioButtons = document.querySelectorAll('input[name="hour_rings"]');
    const hourRingSettingsContainer = document.getElementById('hour-ring-container');
    radioButtons.forEach(radio => {
        radio.addEventListener('change', function () {
            const selectedValue = this.value;
            generateHourRings(selectedValue);  // Generate hour rings based on selected value
            sendMoireData('hour');  // Send hour data
        });
    });

    // Function to generate hour rings based on the selected number
    function generateHourRings(numHourRings) {
        // Clear existing hour rings
        hourRingSettingsContainer.innerHTML = '';

        for (let i = 1; i <= numHourRings; i++) {
            const hourRingDiv = document.createElement('div');
            hourRingDiv.classList.add('form-container');
            hourRingDiv.classList.add('ring-param');
            hourRingDiv.id = `hour_rings_${i}`;

            // Create HTML content for each hour ring
            hourRingDiv.innerHTML = `
                <h2>Hour Ring ${i}</h2>
                <div class="width-input-container" data-target="hour_${i}">
                    <div class="inline-container">
                        <label for="hour_base_width_${i}">Base Width:</label>
                        <input type="range" id="hour_base_width_${i}" name="hour_base_width_${i}" min="0.1" max="1.9" step="0.01">
                        <div class="min-max-label">
                            <span>0.1</span>
                            <span>1.9</span>
                        </div>
                    </div>
                    <div class="inline-container">
                        <label for="hour_revealer_width_${i}">Revealer Width:</label>
                        <input type="range" id="hour_rev_width_${i}" name="hour_rev_width_${i}" min="0.1" max="1.9" step="0.01">
                        <div class="min-max-label">
                            <span>0.1</span>
                            <span>1.9</span>
                        </div>
                    </div>
                </div>
                <label for="hour_ring_angle_${i}">Hour Ring Angle:</label>
                <input type="range" id="hour_ring_angle_${i}" name="hour_ring_angle_${i}" min="-87" max="87" step="1">
                <div class="min-max-label">
                    <span>-87</span>
                    <span>87</span>
                </div>
            `;

            const targetId = `hour_${i}`;
            // Event listeners for mouse enter and leave on hour ring div
            hourRingDiv.addEventListener('mouseenter', () => {
                changePathColor(targetId, 'red');  // Change path color to red on mouse enter
            });
            hourRingDiv.addEventListener('mouseleave', () => {
                changePathColor(targetId, 'black');  // Change path color to black on mouse leave
            });

            // Event listeners for input changes within hour ring div
            hourRingDiv.querySelectorAll('input').forEach(input => {
                input.addEventListener('change', () => {
                    sendMoireData('ring', i+1);
                });
            });

            // Append the hour ring div to the container
            hourRingSettingsContainer.appendChild(hourRingDiv);
        }
    }
}
// Function to load event listener for file input changes
function loadFileInputEvent() {
    document.getElementById('fileInput').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file && file.type === 'image/svg+xml') {
            const reader = new FileReader();
            reader.onload = function(e) {
                const svgElement = svgifyString(e.target.result);
                const complexClip = document.getElementById('complexClip');
                const input_svg = document.getElementById('input_svg');
                const base_svg = document.getElementById('base_svg');
                const viewBox = svgElement.getAttribute("viewBox");
                const [imageX, imageY, imageWidth, imageHeight] = viewBox.split(/[\s,]+/).map(Number);
                base_svg.setAttribute("viewBox", viewBox);
                pasteSvg(svgElement, input_svg);
                pasteSvgContent(svgElement, complexClip);
                rescaleSvg(complexClip, (v => (v - imageX) / imageWidth), (v => (v - imageY) / imageHeight));
                convertToRelative(complexClip);
                handleSvgVisibility(input_svg);
            };
            reader.readAsText(file);
        } else {
            alert('Please upload a valid SVG file.');
            event.target.value = '';  // Clear the input value
        }
    });
}

// Function to load all necessary event listeners
function loadEvents() {
    loadDimensionNumberInputsEvent();  // Load event listeners for dimension number inputs
    loadDimensionToggleViewEvent();    // Load event listeners for dimension toggle view buttons
    loadTabsEvent();                   // Load event listeners for tab buttons
    loadPictureTabsEvent();            // Load event listeners for picture tabs
    loadFileInputEvent();              // Load event listener for file input changes
    loadRingParamEvent();              // Load event listeners for ring parameter changes
    loadDemoControlsEvents();          // Load event listeners for demo controls
}

    //------// DIMENSION GRAPHICS //------//
  // Function to remove an existing circle from the SVG element
function removeCircle(svgElement, inputId) {
    const existingCircle = svgElement.querySelector(`#${inputId}`);
    if (existingCircle) {
        svgElement.removeChild(existingCircle);
    }
}

// Function to add a new circle to the SVG element
function addCircle(svgElement, inputId, radius, color = 'blue') {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '50%');  // Center the circle horizontally
    circle.setAttribute('cy', '50%');  // Center the circle vertically
    circle.setAttribute('r', radius);  // Set the radius of the circle
    circle.setAttribute('fill', 'none');  // No fill color
    circle.setAttribute('stroke', color);  // Set the stroke color
    circle.setAttribute('stroke-width', '2');  // Set the stroke width
    circle.setAttribute('id', inputId);  // Set the ID attribute
    svgElement.appendChild(circle);  // Append the circle to the SVG element
}

// Function to handle changes in the diameter input
function handleDiameterChange(inputId, color = 'blue') {
    const input = document.getElementById(inputId);
    const toggleView = input.nextElementSibling;  // Get the button next to the input
    const diameter = parseFloat(input.value);
    const svgElement = document.getElementById('dimensions_svg');

    // Remove existing circles
    removeCircle(svgElement, inputId);

    if (toggleView.textContent === 'View') {
        if (!isNaN(diameter) && diameter > 0) {
            addCircle(svgElement, inputId, diameter / 2, color);  // Add a new circle with the specified diameter
        } else {
            alert('Please enter a valid diameter.');
        }
    }
    loaded = false;
}

// Function to toggle the view of the circle
function toggleView(inputId, color = 'blue') {
    const input = document.getElementById(inputId);
    const button = input.nextElementSibling;  // Get the button next to the input

    if (button.textContent === 'View') {
        button.textContent = 'Hide';  // Change button text to 'Hide'
    } else {
        button.textContent = 'View';  // Change button text to 'View'
        if (!isSvgVisible(document.getElementById('dimensions_svg'))) {
            document.getElementById('dimension_button').click();  // Click the dimension button if SVG is not visible
        }
    }
    handleDiameterChange(inputId, color);  // Handle the diameter change
}

    //------// SITE GRAPHICS //------//
  // Function to load the demo SVG elements and make them visible
function loadDemo() {
    const revealer = document.getElementById('revealer_svg');
    revealer.classList.add('rotate');  // Add rotation class to the revealer SVG
    handleSvgVisibility(revealer);  // Handle visibility of the revealer SVG
    const base = document.getElementById('base_svg');
    showSVG(base);  // Show the base SVG
}

//------// SVG AND PATH MANIPULATIONS //------//

// Function to convert a string into an SVG element
function svgifyString(input) {
    const parser = new DOMParser();
    const svgDoc = parser.parseFromString(input, 'image/svg+xml');
    return svgDoc.documentElement;
}

// Function to replace SVG content by ID from source to destination
function replaceSvgContentById(id, src, dest) {
    const srcElement = src.getElementById(id);
    const destElement = dest.getElementById(id);
    if (srcElement && destElement) {
        const clonedElement = srcElement.cloneNode(true);
        destElement.parentNode.replaceChild(clonedElement, destElement);
    } else {
        console.error(`Element with id "${id}" not found in both source and destination.`);
    }
}

// Function to paste SVG content from source to destination
function pasteSvgContent(src, dest) {
    dest.innerHTML = '';  // Clear the destination content
    addSvgContent(src, dest);  // Add new content from the source
}

// Function to add SVG content from source to destination
function addSvgContent(src, dest) {
    Array.from(src.childNodes).forEach(child => {
        dest.appendChild(child.cloneNode(true));  // Clone and append each child node
    });
}

// Function to paste an entire SVG element from source to destination
function pasteSvg(src, dest) {
    const viewBox = src.getAttribute('viewBox');
    if (viewBox) {
        dest.setAttribute('viewBox', viewBox);
    } else {
        const width = src.getAttribute('width').replace(/[^\d.]/g, '') || 200;
        const height = src.getAttribute('height').replace(/[^\d.]/g, '') || 200;
        dest.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
    dest.setAttribute('width', '100%');
    dest.setAttribute('height', '100%');
    // Clear previous SVG content
    pasteSvgContent(src, dest);
}

// Function to check if an SVG element is visible
function isSvgVisible(svgElement) {
    return svgElement.classList.contains('active');
}

// Function to show an SVG element
function showSVG(svgElement) {
    svgElement.classList.add('active');
}

// Function to hide an SVG element
function hideSVG(svgElement) {
    svgElement.classList.remove('active');
}

// Function to handle the visibility of an SVG element
function handleSvgVisibility(svgElement) {
    if (svgElement.getAttribute('id') === 'dimensions_svg') {
        if (isSvgVisible(svgElement)) {
            hideSVG(svgElement);
            return false;
        } else {
            showSVG(svgElement);
        }
    } else {
        document.querySelectorAll('.svg-container svg').forEach(svg => {
            if (svg.getAttribute('id') !== 'dimensions_svg') {
                hideSVG(svg);
            }
        });
        showSVG(svgElement);
        const dim_svg = document.getElementById('dimensions_svg');
        const viewBox = svgElement.getAttribute('viewBox');
        dim_svg.setAttribute('viewBox', viewBox);
    }
    return true;
}
// Function to change the color of a specific path element by its ID
function changePathColor(pathId, color) {
    document.getElementById('svg_container').querySelectorAll(`#${pathId}`).forEach(path => {
        path.setAttribute('fill', color);
    });
}

//------// SERVER RESPONSE HANDLERS  //-------//

// Function to replace SVG content based on info_data
function replaceSvgContentByIdRevAndBase(info_data, baseSvg, revealerSvg){
    const svgB = document.getElementById('base_svg');
    const svgR = document.getElementById('revealer_svg');
    let id = '';
    switch (info_data){
        case 0:
            id = 'outer';
            break;
        case 1:
            id = 'minute';
            break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            id = `hour_${info_data-1}`;
            break;
        default:
    }
    replaceSvgContentById(id, baseSvg, svgB);
    replaceSvgContentById(id, revealerSvg, svgR);
}

// Function to remove hour paths from base and revealer SVGs
function removeHoursRevAndBase() {
    let svgB = document.getElementById('base_svg');
    let svgR = document.getElementById('revealer_svg');
    svgB.querySelectorAll('path').forEach(path => {
        if (path.id.includes('hour')) {
            path.remove();
        }
    });
    svgR.querySelectorAll('path').forEach(path => {
        if (path.id.includes('hour')) {
            path.remove();
        }
    });
}

// Function to paste hour paths into base and revealer SVGs
function pasteHourRevealerAndBase(baseSvg, revealerSvg) {
    const svgB = document.getElementById('base_svg');
    const svgR = document.getElementById('revealer_svg');
    addSvgContent(baseSvg, svgB);
    addSvgContent(revealerSvg, svgR);
}

// Function to paste base and revealer SVGs and adjust clipPath
function pasteRevealerAndBase(baseSvg, revealerSvg) {
    const svgB = document.getElementById('base_svg');
    const svgR = document.getElementById('revealer_svg');
    const clipPath = svgB.getElementById("complexClip");
    const [imageX_i, imageY_i, imageWidth_i, imageHeight_i] = svgB.getAttribute("viewBox").split(/[\s,]+/).map(Number);
    pasteSvg(baseSvg, svgB);
    viewBox = svgB.getAttribute('viewBox');
    const [imageX_f, imageY_f, imageWidth_f, imageHeight_f] = svgB.getAttribute("viewBox").split(/[\s,]+/).map(Number);
    pasteSvg(revealerSvg, svgR);
    rescaleSvg(clipPath, v => (v * imageWidth_i / imageWidth_f), v => (v * imageHeight_i / imageHeight_f));
    translateRelativeSvg(clipPath, (imageWidth_f - imageWidth_i) / (2 * imageWidth_f), (imageHeight_f - imageHeight_i) / (2 * imageHeight_f))
    svgB.querySelector('defs').appendChild(clipPath);
}

//------// SERVER AND FORM FUNCTIONS //-------//

// Function to validate form data
function valid_data(form){
    const form_data = new FormData(form);
    let inner_radius = form_data.get("inner_diameter");
    let middle_radius = form_data.get("middle_diameter");
    let outer_radius = form_data.get("outer_diameter");
    let inner_hour_bound = form_data.get("inner_hour_diameter");
    let outer_hour_bound =  form_data.get("outer_hour_diameter");
    if(!inner_radius || !inner_hour_bound || !outer_hour_bound || !middle_radius || !outer_radius){
        alert("please fill in dimensions and make sure they are in a ascending order");
        return false;
    }
    inner_radius = parseFloat(inner_radius);
    middle_radius = parseFloat(middle_radius);
    outer_radius = parseFloat(outer_radius);
    inner_hour_bound = parseFloat(inner_hour_bound);
    outer_hour_bound = parseFloat(outer_hour_bound);
    if(inner_hour_bound <= inner_radius || outer_hour_bound <= inner_hour_bound ||middle_radius <= outer_hour_bound || outer_radius <= middle_radius){
        alert("please check given dimensions make sure they are in a ascending order");
        return false;
    }
    return !(!form_data.get("hour_rings") || !form_data.get("speed_ratio"));
}
// Function to send Moire data to the server and handle the response
async function sendMoireData(info, info_data = '') {
    console.log(info);
    const url = 'http://127.0.0.1:5000/process_data';
    const form = document.getElementById('mainForm');

    // If info is not 'full' and data is not loaded, set info to 'grid'
    if(info !== 'full' && !loaded){
        info = 'grid';
        info_data = '';
    }

    // Validate form data
    if(!valid_data(form)) return;

    // Get the value of the input element
    const form_data = new FormData(form);
    let rings = form_data.get("hour_rings")/1 + 2;
    let rings_radii = [form_data.get("outer_diameter")/2 ,  form_data.get("white_outer_diameter")/2,
                                form_data.get("middle_diameter") / 2, form_data.get("outer_hour_diameter")/2,
                                form_data.get("inner_hour_diameter")/2, form_data.get("inner_diameter")/2];
    let rev_width = new Array(rings);
    let base_width = new Array(rings);
    let base_alphas = new Array(rings);

    // Set initial values for minute and outer rings
    rev_width[0] = form_data.get("outer_rev_width")/1;
    base_width[0] = form_data.get("outer_base_width")/1;
    base_alphas[0] = form_data.get("outer_ring_angle") /1;
    rev_width[1] = form_data.get("minute_rev_width")/1;
    base_width[1] = form_data.get("minute_base_width")/1;
    base_alphas[1] = form_data.get("minute_ring_angle") /1;


    // Loop through hour rings and set their values
    for (let i = 2; i < rings; i++) {
        console.log(`hour_rev_width_${i-1}`)
        rev_width[i] = form_data.get(`hour_rev_width_${i-1}`)/1;
        base_width[i] = form_data.get(`hour_base_width_${i-1}`)/1;
        base_alphas[i] = form_data.get(`hour_ring_angle_${i-1}`) /1;
    }

    const speed = form_data.get("speed_ratio")/1;
    const svgElement = document.getElementById('input_svg');

    // Create data object to send to the server
    const data = {
        rings_radii: rings_radii,
        rings: rings,
        speed: speed,
        rev_width: rev_width,
        base_width: base_width,
        base_alphas: base_alphas,
        info: info,
        info_data: info_data,
        svg_content: svgElement.outerHTML,
        svg_viewbox: svgElement.getAttribute('viewBox')
    };

    console.log(data);

    try {
        // Send data to the server
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        // Check if the response is ok
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Response from server:', result);

        const baseSvg = svgifyString(result.base);
        const revealerSvg = svgifyString(result.revealer);

        // Handle different info types
        if(info === 'grid'){
            setRotationDegrees(-360/speed)
            pasteRevealerAndBase(baseSvg, revealerSvg);
            loaded = true;
        } else if(info === 'hour'){
            removeHoursRevAndBase();
            pasteHourRevealerAndBase(baseSvg, revealerSvg);
        } else if(info === 'ring'){
            replaceSvgContentByIdRevAndBase(info_data, baseSvg, revealerSvg);
        }
        return result;
    } catch (error) {
        console.error('Error:', error);
    }
}

//-------// THESE ARE FUNCTIONS FOR FITTING THE CLIP-PATHS //-------//

// Function to convert absolute path commands to relative path commands in an SVG element
function convertToRelative(svgElement) {
    const paths = svgElement.querySelectorAll('path');
    paths.forEach(pathElement => {
        const pathData = pathElement.getAttribute('d');
        if (!pathData) return;
        const commands = pathData.match(/[a-z][^a-z]*/ig);
        let currentX = 0, currentY = 0;
        let startX = 0, startY = 0;
        const relativeCommands = commands.map(command => {
            const type = command[0];
            const values = command.slice(1).trim().split(/[\s,]+/).map(Number);
            switch (type) {
                case 'M': // Move to
                    const relM = values.map((v, i) => (i % 2 === 0 ? v - currentX : v - currentY));
                    startX = values[0];
                    startY = values[1];
                    currentX = values[0];
                    currentY = values[1];
                    return `m${relM.join(' ')}`;
                case 'L': // Line to
                    const relX = values[0] - currentX;
                    const relY = values[1] - currentY;
                    currentX = values[0];
                    currentY = values[1];
                    return `l${relX} ${relY}`;
                case 'H': // Horizontal line to
                    const relHX = values[0] - currentX;
                    currentX = values[0];
                    return `h${relHX}`;
                case 'V': // Vertical line to
                    const relVY = values[0] - currentY;
                    currentY = values[0];
                    return `v${relVY}`;
                case 'C': // Cubic Bezier curve
                    const relC = values.map((v, i) => (i % 2 === 0 ? v - currentX : v - currentY));
                    currentX = values[4];
                    currentY = values[5];
                    return `c${relC.join(' ')}`;
                case 'S': // Smooth cubic Bezier curve
                    const relS = values.map((v, i) => (i % 2 === 0 ? v - currentX : v - currentY));
                    currentX = values[2];
                    currentY = values[3];
                    return `s${relS.join(' ')}`;
                case 'Q': // Quadratic Bezier curve
                    const relQ = values.map((v, i) => (i % 2 === 0 ? v - currentX : v - currentY));
                    currentX = values[2];
                    currentY = values[3];
                    return `q${relQ.join(' ')}`;
                case 'T': // Smooth quadratic Bezier curve
                    const relT = values.map((v, i) => (i % 2 === 0 ? v - currentX : v - currentY));
                    currentX = values[0];
                    currentY = values[1];
                    return `t${relT.join(' ')}`;
                case 'A': // Arc to
                    const relA = values.slice(0, 5).concat(values[5] - currentX, values[6] - currentY);
                    currentX = values[5];
                    currentY = values[6];
                    return `a${relA.join(' ')}`;
                case 'Z': // Close path
                    currentX = startX;
                    currentY = startY;
                    return 'z';
                default:
                    return command;
            }
        });
        pathElement.setAttribute('d', relativeCommands.join(' '));
    });
}
// Function to rescale the SVG paths based on provided transformation functions for x and y coordinates
function rescaleSvg(svgElement, transform_x, transform_y) {
    const paths = svgElement.querySelectorAll('path');
    paths.forEach(pathElement => {
        const pathData = pathElement.getAttribute('d');
        if (!pathData) return;
        const commands = pathData.match(/[a-z][^a-z]*/ig);
        const normalizedCommands = commands.map(command => {
            const type = command[0];
            let values = command.slice(1).trim();
            // Handle shorthand notation for decimal numbers
            values = values.replace(/(\s|,|^)\./g, '$10.');

            // Split values into an array of numbers, handling negative signs and decimals
            if (values)
                values = values.match(/-?\d*\.?\d+(?:e[+-]?\d+)?/g).map(Number);
            switch (type.toLowerCase()) {
                case 'm': // Move to
                case 'l': // Line to
                case 't': // Smooth quadratic Bezier curve to
                    return `${type} ${values.map((v, i) => (i % 2 === 0 ? transform_x(v) : transform_y(v))).join(' ')}`;
                case 'h': // Horizontal line to
                    return `${type} ${values.map(v => transform_x(v)).join(' ')}`;
                case 'v': // Vertical line to
                    return `${type} ${values.map(v => transform_y(v)).join(' ')}`;
                case 'c': // Cubic Bezier curve to
                case 's': // Smooth cubic Bezier curve to
                case 'q': // Quadratic Bezier curve to
                case 'a': // Arc to
                    return `${type} ${values.map((v, i) => (i % 2 === 0 ? transform_x(v) : transform_y(v))).join(' ')}`;
                case 'z': // Close path
                    return type;
                default:
                    return command;
            }
        });
        pathElement.setAttribute('d', normalizedCommands.join(' '));
    });
}

// Function to translate the SVG paths by a given dx and dy
function translateRelativeSvg(svgElement, dx, dy) {
    const paths = svgElement.querySelectorAll('path');
    paths.forEach(pathElement => {
        const pathData = pathElement.getAttribute('d');
        if (!pathData) return;
        const commands = pathData.match(/[a-z][^a-z]*/ig);
        let firstMove = true;
        const translatedCommands = commands.map(command => {
            const type = command[0];
            let values = command.slice(1).trim();
            // Handle shorthand notation for decimal numbers
            values = values.replace(/(\s|,|^)\./g, '$10.');

            // Split values into an array of numbers, handling negative signs and decimals
            if (values)
                values = values.match(/-?\d*\.?\d+(?:e[+-]?\d+)?/g).map(Number);

            if (type.toLowerCase() === 'm' && firstMove) {
                // Apply translation to the first 'M' command
                firstMove = false;
                return `${type} ${values.map((v, i) => (i % 2 === 0 ? v + dx : v + dy)).join(' ')}`;
            } else {
                // Return the command unchanged
                return command;
            }
        });
        pathElement.setAttribute('d', translatedCommands.join(' '));
    });
}
