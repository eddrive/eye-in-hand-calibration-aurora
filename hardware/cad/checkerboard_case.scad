// Case per foglio A6 con braccetto e pocket triangolari
// === PARAMETRI MODIFICABILI ===
// Dimensioni foglio A6 (spazio interno utile)
a6_width = 105;
a6_height = 148;
base_thickness = 3;

// Bordino perimetrale
border_height = 4;       // Altezza del bordino
border_width = 2.5;      // Larghezza del bordino

// Pocket triangolari agli angoli
pocket_size = 12;        // Lato del triangolo
pocket_thickness = 2;  // Spessore dell'aletta triangolare
pocket_gap = 2;        // Distanza dalla sommità del bordino

// Braccetto e pin
pin_diameter = 6;
pin_height = 15;
pin_spacing = 20;
pin_offset = 5;          // Distanza dei pin dall'estremità libera del braccetto
arm_width = 40;
arm_thickness = 5;
arm_length = 30;

// Risoluzione cilindri
cylinder_resolution = 64;

// === MODULI ===

// Base del foglio A6 con bordino perimetrale
module a6_base() {
    // Dimensioni totali della base (foglio + bordini)
    total_width = a6_width + 2 * border_width;
    total_height = a6_height + 2 * border_width;
    
    // Base completa
    cube([total_width, total_height, base_thickness]);
    
    // Bordino perimetrale
    // Lato sinistro
    translate([0, 0, base_thickness])
        cube([border_width, total_height, border_height]);
    
    // Lato destro
    translate([total_width - border_width, 0, base_thickness])
        cube([border_width, total_height, border_height]);
    
    // Lato inferiore
    translate([0, 0, base_thickness])
        cube([total_width, border_width, border_height]);
    
    // Lato superiore
    translate([0, total_height - border_width, base_thickness])
        cube([total_width, border_width, border_height]);
}

// Pocket triangolare per tenere il foglio
module corner_pocket_triangle() {
    // Triangolo sottile che parte dall'altezza del bordino
    translate([0, 0, base_thickness + border_height - pocket_gap])
        linear_extrude(height = pocket_thickness)
            polygon(points = [[0, 0], [pocket_size, 0], [0, pocket_size]]);
}

// Braccetto con due pin sopra (spostati verso l'estremità libera)
module pin_arm() {
    cube([arm_width, arm_length, arm_thickness]);
    y_pin = pin_diameter/2 + pin_offset;
    
    // Pin sinistro
    translate([arm_width/2 - pin_spacing/2, y_pin, arm_thickness])
        cylinder(h=pin_height, d=pin_diameter, $fn=cylinder_resolution);
    
    // Pin destro
    translate([arm_width/2 + pin_spacing/2, y_pin, arm_thickness])
        cylinder(h=pin_height, d=pin_diameter, $fn=cylinder_resolution);
}

// === ASSEMBLAGGIO ===

module a6_case_with_arm() {
    // Dimensioni totali della base (foglio + bordini)
    total_width = a6_width + 2 * border_width;
    total_height = a6_height + 2 * border_width;
    
    union() {
        // Base
        a6_base();
        
        // 4 pocket triangolari agli angoli
        // In basso a sinistra
        translate([border_width, border_width, 0])
            corner_pocket_triangle();
        
        // In basso a destra
        translate([total_width - border_width, border_width, 0])
            rotate([0, 0, 90])
            corner_pocket_triangle();
        
        // In alto a sinistra
        translate([border_width, total_height - border_width, 0])
            rotate([0, 0, -90])
            corner_pocket_triangle();
        
        // In alto a destra
        translate([total_width - border_width, total_height - border_width, 0])
            rotate([0, 0, 180])
            corner_pocket_triangle();
        
        // Braccetto centrato sul lato corto in basso (lato libero)
        translate([total_width/2 - arm_width/2, -arm_length, 0])
            pin_arm();
    }
}

// === OUTPUT ===
a6_case_with_arm();