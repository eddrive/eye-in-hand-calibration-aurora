// Case per Generatore Campo EM Aurora con braccetto compatto e pin leggermente arretrati

// === PARAMETRI MODIFICABILI ===
generator_length = 200;
generator_width = 71;
generator_height = 200;

case_height = 50;
wall_thickness = 3;
clearance = 1;

// Braccetto e pin
pin_diameter = 6;
pin_height = 15;
pin_spacing = 20;        // distanza centro-centro
arm_width = 40;          // larghezza
arm_thickness = 5;       // spessore
arm_length = 30;         // lunghezza

// === MODULI ===

// Case principale
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

// Braccetto con due pin sopra (leggermente arretrati)
module pin_arm() {
    // Base del braccetto
    cube([arm_width, arm_length, arm_thickness]);

    // Pin arretrati di 5 mm rispetto all’estremità
    y_pin = arm_length - pin_diameter/2 - 5;

    // Pin sinistro
    translate([arm_width/2 - pin_spacing/2, y_pin, arm_thickness])
    cylinder(h=pin_height, d=pin_diameter, $fn=64);

    // Pin destro
    translate([arm_width/2 + pin_spacing/2, y_pin, arm_thickness])
    cylinder(h=pin_height, d=pin_diameter, $fn=64);
}

// === ASSEMBLAGGIO ===
module assembled_case() {
    union() {
        generator_case();

        // Braccetto centrato sul lato lungo del case
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
