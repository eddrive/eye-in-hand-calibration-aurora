// A6 sheet holder with mounting arm and corner pockets

// === PARAMETERS ===

// A6 sheet dimensions (internal space)
a6_width = 105;
a6_height = 148;
base_thickness = 3;

// Perimeter border
border_height = 4;
border_width = 2.5;

// Corner pockets to hold sheet
pocket_size = 12;        // triangle side length
pocket_thickness = 2;
pocket_gap = 2;          // distance from border top

// Mounting arm and pins
pin_diameter = 6;
pin_height = 15;
pin_spacing = 20;
pin_offset = 5;          // pin distance from arm free end
arm_width = 40;
arm_thickness = 5;
arm_length = 30;

$fn = 64;

// === MODULES ===

module a6_base() {
    total_width = a6_width + 2 * border_width;
    total_height = a6_height + 2 * border_width;
    
    cube([total_width, total_height, base_thickness]);
    
    // Left border
    translate([0, 0, base_thickness])
        cube([border_width, total_height, border_height]);
    
    // Right border
    translate([total_width - border_width, 0, base_thickness])
        cube([border_width, total_height, border_height]);
    
    // Bottom border
    translate([0, 0, base_thickness])
        cube([total_width, border_width, border_height]);
    
    // Top border
    translate([0, total_height - border_width, base_thickness])
        cube([total_width, border_width, border_height]);
}

module corner_pocket_triangle() {
    translate([0, 0, base_thickness + border_height - pocket_gap])
        linear_extrude(height = pocket_thickness)
            polygon(points = [[0, 0], [pocket_size, 0], [0, pocket_size]]);
}

module pin_arm() {
    cube([arm_width, arm_length, arm_thickness]);
    y_pin = pin_diameter/2 + pin_offset;
    
    translate([arm_width/2 - pin_spacing/2, y_pin, arm_thickness])
        cylinder(h=pin_height, d=pin_diameter, $fn=64);
    
    translate([arm_width/2 + pin_spacing/2, y_pin, arm_thickness])
        cylinder(h=pin_height, d=pin_diameter, $fn=64);
}

module a6_case_with_arm() {
    total_width = a6_width + 2 * border_width;
    total_height = a6_height + 2 * border_width;
    
    union() {
        a6_base();
        
        // Four corner pockets
        translate([border_width, border_width, 0])
            corner_pocket_triangle();
        
        translate([total_width - border_width, border_width, 0])
            rotate([0, 0, 90])
            corner_pocket_triangle();
        
        translate([border_width, total_height - border_width, 0])
            rotate([0, 0, -90])
            corner_pocket_triangle();
        
        translate([total_width - border_width, total_height - border_width, 0])
            rotate([0, 0, 180])
            corner_pocket_triangle();
        
        // Mounting arm centered on bottom short side
        translate([total_width/2 - arm_width/2, -arm_length, 0])
            pin_arm();
    }
}

// === OUTPUT ===
a6_case_with_arm();