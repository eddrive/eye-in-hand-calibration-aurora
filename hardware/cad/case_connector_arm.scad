// Braccetto di collegamento a S tra due case (2D sul piano XY)

// === PARAMETRI MODIFICABILI ===

// Parametri pin
pin_diameter = 6;
pin_spacing = 20;
pin_offset = 5;          // Distanza dei pin dall'estremità del mini braccetto
pin_clearance = 0.3;     // Tolleranza per inserimento facile

// Forma a S (piana)
s_offset = 10;           // Spostamento avanti del primo e terzo tratto (mm)
s_lateral = 100;         // Lunghezza del tratto orizzontale (mm)
connector_width = 20;    // Larghezza del corpo (mm)
connector_thickness = 10; // Spessore del corpo (Z)

// Estremità per incastro
step_width = 40;         // Larghezza delle estremità
step_length = 30;        // Lunghezza delle estremità
step_thickness = 5;      // Spessore delle estremità
step_clearance = 0.3;    // Gioco per incastro

// Profondità fori
hole_depth = 16;         // Profondità dei fori per i pin

// Risoluzione
$fn = 64;

// === MODULI ===

// Fori per estremità inferiore
module pin_holes_bottom() {
    pin_position = step_length - (pin_diameter/2 + pin_offset);
    translate([step_width/2 - pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
    translate([step_width/2 + pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
}

// Fori per estremità superiore
module pin_holes_top() {
    pin_position = pin_diameter/2 + pin_offset;
    translate([step_width/2 - pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
    translate([step_width/2 + pin_spacing/2, pin_position, -0.1])
        cylinder(h=hole_depth + 0.2, d=pin_diameter + pin_clearance);
}

// Braccetto completo a S piana
module connector_arm() {
    difference() {
        union() {
            // === ESTREMITÀ INFERIORE ===
            translate([(connector_width - step_width)/2, 0, 0])
                cube([step_width - step_clearance, step_length, step_thickness]);

            // === PRIMO TRATTO: 10mm avanti (con sovrapposizione) ===
            translate([0, step_length - 0.1, 0])
                cube([connector_width, s_offset + 0.1, connector_thickness]);

            // === SECONDO TRATTO: 100mm orizzontale (con sovrapposizione) ===
            translate([0, step_length + s_offset - 0.1, 0])
                cube([s_lateral + 0.1, connector_width, connector_thickness]);

            // === TERZO TRATTO: 10mm avanti (CORRETTO - inizia dove finisce il secondo) ===
            translate([s_lateral-(step_width/2), step_length + s_offset + connector_width - 0.1, 0])
                cube([connector_width + 0.1, s_offset + 0.1, connector_thickness]);

            // === ESTREMITÀ SUPERIORE (con sovrapposizione) ===
            translate([s_lateral + (connector_width - step_width)/2 - (step_width/2), step_length + 2*s_offset + connector_width - 0.1, 0])
                cube([step_width - step_clearance, step_length + 0.1, step_thickness]);
        }

        // Fori estremità inferiore
        translate([(connector_width - step_width)/2, 0, -0.1])
            pin_holes_bottom();

        // Fori estremità superiore
        translate([s_lateral + (connector_width - step_width)/2 - (step_width/2), step_length + 2*s_offset + connector_width, -0.1])
            pin_holes_top();
    }
}

// === OUTPUT ===
connector_arm();