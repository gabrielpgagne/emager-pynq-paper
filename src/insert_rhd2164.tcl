set RHD2164_SPI_ROOT "$$RHD2164_SPI_ROOT"
set PWD [get_property DIRECTORY [current_project]]

update_compile_order -fileset sources_1

import_files -fileset constrs_1 "$RHD2164_SPI_ROOT/vivado/$BOARD/rhd2164-spi.xdc"
import_files -norecurse [list "$RHD2164_SPI_ROOT/hdl/spi_master_cs.v" "$RHD2164_SPI_ROOT/hdl/rhd_wrapper.v" "$RHD2164_SPI_ROOT/hdl/spi_master.v"]
set_property target_constrs_file "$PWD/finn_zynq_link.srcs/constrs_1/imports/$BOARD/rhd2164-spi.xdc" [current_fileset -constrset]
update_compile_order -fileset sources_1

create_bd_cell -type module -reference rhd_wrapper rhd_wrapper_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_cfg
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_status
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_data

set_property -dict [list CONFIG.C_GPIO_WIDTH {24} CONFIG.C_ALL_OUTPUTS {1}] [get_bd_cells axi_gpio_cfg]
set_property -dict [list CONFIG.C_GPIO_WIDTH {1} CONFIG.C_GPIO2_WIDTH {1} CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_ALL_OUTPUTS_2 {0}] [get_bd_cells axi_gpio_status]
set_property -dict [list CONFIG.C_GPIO_WIDTH {16} CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_ALL_OUTPUTS_2 {0}] [get_bd_cells axi_gpio_data]

connect_bd_net [get_bd_pins axi_gpio_cfg/gpio_io_o] [get_bd_pins rhd_wrapper_0/i_ctrl]
connect_bd_net [get_bd_pins axi_gpio_status/gpio_io_o] [get_bd_pins rhd_wrapper_0/i_start]
connect_bd_net [get_bd_pins axi_gpio_status/gpio2_io_i] [get_bd_pins rhd_wrapper_0/o_done]
connect_bd_net [get_bd_pins axi_gpio_data/gpio2_io_i] [get_bd_pins rhd_wrapper_0/o_dout]
connect_bd_net [get_bd_pins axi_gpio_data/gpio_io_o] [get_bd_pins rhd_wrapper_0/i_din]

create_bd_port -dir I i_miso
create_bd_port -dir O o_mosi
create_bd_port -dir O o_cs
create_bd_port -dir O o_sclk

connect_bd_net [get_bd_ports o_sclk] [get_bd_pins rhd_wrapper_0/o_sclk]
connect_bd_net [get_bd_ports o_mosi] [get_bd_pins rhd_wrapper_0/o_mosi]
connect_bd_net [get_bd_ports o_cs] [get_bd_pins rhd_wrapper_0/o_cs]
connect_bd_net [get_bd_ports i_miso] [get_bd_pins rhd_wrapper_0/i_miso]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ps/FCLK_CLK0 (100 MHz)} Clk_slave {Auto} Clk_xbar {/zynq_ps/FCLK_CLK0 (100 MHz)} Master {/zynq_ps/M_AXI_GP0} Slave {/axi_gpio_cfg/S_AXI} ddr_seg {Auto} intc_ip {/axi_interconnect_0} master_apm {0}}  [get_bd_intf_pins axi_gpio_cfg/S_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ps/FCLK_CLK0 (100 MHz)} Clk_slave {Auto} Clk_xbar {/zynq_ps/FCLK_CLK0 (100 MHz)} Master {/zynq_ps/M_AXI_GP0} Slave {/axi_gpio_status/S_AXI} ddr_seg {Auto} intc_ip {/axi_interconnect_0} master_apm {0}}  [get_bd_intf_pins axi_gpio_status/S_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ps/FCLK_CLK0 (100 MHz)} Clk_slave {Auto} Clk_xbar {/zynq_ps/FCLK_CLK0 (100 MHz)} Master {/zynq_ps/M_AXI_GP0} Slave {/axi_gpio_data/S_AXI} ddr_seg {Auto} intc_ip {/axi_interconnect_0} master_apm {0}}  [get_bd_intf_pins axi_gpio_data/S_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins rhd_wrapper_0/i_clk]

# Force regenerate the top wrapper
make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top
add_files -norecurse "$PWD/finn_zynq_link.gen/sources_1/bd/top/hdl/top_wrapper.v"

regenerate_bd_layout
save_bd_design
validate_bd_design -force
update_compile_order -fileset sources_1
set_property top top_wrapper [current_fileset]
save_bd_design