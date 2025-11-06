// S-shaped connector arm between enclosures

// === PARAMETERS ===

// Pin specifications
pin_diameter = 6;
pin_spacing = 20;
pin_offset = 5;
pin_clearance = 0.3;

// S-curve geometry
s_offset = 10;           // forward offset of first and third segments
s_lateral = 100;         // horizontal segment length
connector_width = 20;
connector_thickness = 10;

// End pieces for interlocking
step_width = 40;
step_length = 30;
step_thickness = 5;
step_clearance = 0.3;

hole_depth = 16;

$fn = 64;

// === MODULES ===

module pin_holes_bottom() {
    pin_position = step_length - (pin_diameter/2 + pin_offset);
    translate([step_width/2 - pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
    translate([step_width/2 + pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
}

module pin_holes_top() {
    pin_position = pin_diameter/2 + pin_offset;
    translate([step_width/2 - pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
    translate([step_width/2 + pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
}

module connector_arm() {
    difference() {
        union() {
            // Bottom end piece
            translate([(connector_width - step_width)/2, 0, 0])
                cube([step_width - step_clearance, step_length, step_thickness]);

            // First segment: forward offset with overlap
            translate([0, step_length - 0.1, 0])
                cube([connector_width, s_offset + 0.1, connector_thickness]);

            // Second segment: horizontal span with overlap
            translate([0, step_length + s_offset - 0.1, 0])
                cube([s_lateral + 0.1, connector_width, connector_thickness]);

            // Third segment: forward offset
            translate([s_lateral-(step_width/2), step_length + s_offset + connector_width - 0.1, 0])
                cube([connector_width + 0.1, s_offset + 0.1, connector_thickness]);

            // Top end piece with overlap
            translate([s_lateral + (connector_width - step_width)/2 - (step_width/2), step_length + 2*s_offset + connector_width - 0.1, 0])
                cube([step_width - step_clearance, step_length + 0.1, step_thickness]);
        }

        // Bottom pin holes
        translate([(connector_width - step_width)/2, 0, -0.1])
            pin_holes_bottom();

        // Top pin holes
        translate([s_lateral + (connector_width - step_width)/2 - (step_width/2), step_length + 2*s_offset + connector_width, -0.1])
            pin_holes_top();
    }
}

// === OUTPUT ===
connector_arm();