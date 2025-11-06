// EM Field Generator Aurora - Enclosure with mounting arm

// === PARAMETERS ===
generator_length = 200;
generator_width = 71;
generator_height = 200;

case_height = 50;
wall_thickness = 3;
clearance = 1;

// Mounting arm and pins
pin_diameter = 6;
pin_height = 15;
pin_spacing = 20;        // center-to-center distance
arm_width = 40;
arm_thickness = 5;
arm_length = 30;

// === MODULES ===

module generator_case() {
    difference() {
        cube([
            generator_length + 2*wall_thickness + 2*clearance,
            generator_width + 2*wall_thickness + 2*clearance,
            case_height
        ]);
        translate([wall_thickness + clearance, 
                   wall_thickness + clearance, 
                   wall_thickness])
        cube([generator_length, generator_width, case_height]);
    }
}

module pin_arm() {
    cube([arm_width, arm_length, arm_thickness]);

    // Pins offset 5mm from edge
    y_pin = arm_length - pin_diameter/2 - 5;

    translate([arm_width/2 - pin_spacing/2, y_pin, arm_thickness])
    cylinder(h=pin_height, d=pin_diameter, $fn=64);

    translate([arm_width/2 + pin_spacing/2, y_pin, arm_thickness])
    cylinder(h=pin_height, d=pin_diameter, $fn=64);
}

module assembled_case() {
    union() {
        generator_case();

        // Center arm on long side
        translate([
            (generator_length + 2*wall_thickness + 2*clearance)/2 - arm_width/2,
            generator_width + 2*wall_thickness + 2*clearance,
            0
        ])
        pin_arm();
    }
}

// === OUTPUT ===
assembled_case();